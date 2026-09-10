// mikumux.c — MIKU Lossless Video muxer for FFmpeg
//
// This muxer receives already-encoded packets (WebP video frames, FLAC
// audio frames) and packages them into the .miku container. It does NOT do
// any encoding itself -- that's libavcodec's job (the libwebp encoder for
// video, the native flac encoder for audio). See the reply for why that
// division of labor is also the answer to the threading question.
//
// v1 is WebP-only. PNG support existed in earlier versions and has been
// removed: init() now rejects any video stream (primary or thumbnail) whose
// codec isn't AV_CODEC_ID_WEBP.
//
// ---------------------------------------------------------------------
// On-disk layout written by this muxer:
//   [0, 64)   64-byte header (see field offsets below)
//   [64, X)   META block: "META" magic, flags u8, json_len u32, json data,
//             optional thumbnail bytes (present iff flags & MIKU_META_FLAG_THUMBNAIL)
//   [X, Y)    video frame bytes, back to back, one contiguous WebP image
//             per frame (deduplicated -- see miku_write_video_packet)
//   [Y, Z)    audio blob: a complete embedded FLAC stream ("fLaC" + a single
//             STREAMINFO metadata block + raw FLAC frames)
//   [Z, end)  index table: n_frames entries of {offset u64, size u32,
//             duration u32}, followed by exactly one more entry describing
//             the audio blob (offset, size, ~total sample count).
//             Delta mode only (codec_flag == MIKU_CODEC_WEBP_DELTA, header
//             version == MIKU_FORMAT_VERSION_DELTA): each video entry's size
//             field has its MSB stolen as a keyframe flag (pkt->flags &
//             AV_PKT_FLAG_KEY at encode time); the real size is the low 31
//             bits. Off by default -- with delta mode inactive the size
//             field is the plain, full-range byte count exactly as before.
//
// Header field byte offsets (64 bytes total):
//   0   "MIKU" magic (4 bytes)
//   4   version                 u16
//   6   fps_num                 u16   -- matches what mikudec.c already reads
//   8   fps_den                 u16   -- matches what mikudec.c already reads
//   10  frame_count             u32   -- matches what mikudec.c already reads (placeholder, patched in write_trailer)
//   14  sample_rate             u32   -- matches what mikudec.c already reads
//   18  channels                u16   -- matches what mikudec.c already reads
//   20  bit_depth               u16   -- matches what mikudec.c already reads
//   22  metadata_offset         u64   -- always 64 (META block is fixed right after the header)
//   30  index_offset            u64   -- matches what mikudec.c already reads (placeholder, patched in write_trailer)
//   38  reserved (7 bytes)
//   45  codec_flag              u8    -- 0x01 = WebP, the only accepted value in v1 (0x00/PNG is rejected)
//   46  video_offset            u64   -- offset of the first video frame byte
//   54  sar_num                 u16   -- sample aspect ratio numerator, 1 if unset
//   56  sar_den                 u16   -- sample aspect ratio denominator, 1 if unset
//   58  reserved (6 bytes)
//
// NOTE ON THE THUMBNAIL: the spec doesn't give the thumbnail an explicit length
// field. Since it's always the last thing in the META block and video_offset
// marks exactly where it ends, its length is implicit: video_offset - (start of
// thumbnail data). It must be WebP, same as the primary video stream.
// ---------------------------------------------------------------------

#include <string.h>

#include "avformat.h"
#include "mux.h"
#include "packet_internal.h"
#include "flacenc.h"
#include "libavcodec/flac.h"
#include "libavcodec/packet.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/dict.h"
#include "libavutil/mem.h"
#include "libavutil/mathematics.h"

#define MIKU_FORMAT_VERSION    1
#define MIKU_HEADER_SIZE       64

#define MIKU_OFF_VERSION        4
#define MIKU_OFF_FPS_NUM        6
#define MIKU_OFF_FPS_DEN        8
#define MIKU_OFF_FRAME_COUNT    10
#define MIKU_OFF_SAMPLE_RATE    14
#define MIKU_OFF_CHANNELS       18
#define MIKU_OFF_BIT_DEPTH      20
#define MIKU_OFF_METADATA_OFF   22
#define MIKU_OFF_INDEX_OFF      30
#define MIKU_OFF_CODEC_FLAG     45
#define MIKU_OFF_VIDEO_OFF      46
#define MIKU_OFF_SAR_NUM        54
#define MIKU_OFF_SAR_DEN        56

#define MIKU_CODEC_WEBP         0x01
#define MIKU_CODEC_WEBP_DELTA   0x02  // opt-in delta-frame mode (-delta_frames on the
                                       // encoder); index entries carry a keyframe bit,
                                       // see miku_write_video_packet. codec_id on the
                                       // video stream is AV_CODEC_ID_MIKU_DELTA, never
                                       // AV_CODEC_ID_WEBP, when this flag is written.

#define MIKU_FORMAT_VERSION_DELTA 2   // written instead of MIKU_FORMAT_VERSION when
                                       // MIKU_CODEC_WEBP_DELTA is active. A file written
                                       // with delta_frames=0 (the default) is byte-for-
                                       // byte the same header/index shape FFmpeg has
                                       // always produced -- this only ever appears when
                                       // delta mode was explicitly requested.

#define MIKU_META_FLAG_THUMBNAIL 0x02
#define MIKU_META_FLAG_COLOR     0x04

typedef struct MikuIndexEntry {
    int64_t  offset;
    uint32_t size;
    uint32_t duration;
} MikuIndexEntry;

typedef struct MikuMuxContext {
    int video_stream_idx;   // -1 until resolved in init()
    int audio_stream_idx;
    int thumb_stream_idx;   // -1 if no attached-pic stream was provided

    // Resolved once in init() from the video stream's codec_id: 1 iff it's
    // AV_CODEC_ID_MIKU_DELTA (the encoder only ever emits that id when its
    // own -delta_frames option was actually turned on). Gates every format
    // difference below (index entry shape, header version, codec_flag) so
    // that delta_frames=0 output is provably identical, byte for byte, to
    // what this muxer has always produced.
    int delta_mode;

    int waiting_thumb;      // 1 until the thumbnail packet arrives (or write_header
                             // decides there isn't one coming)
    int header_written;     // 1 once the 64-byte header + META block are on disk
    PacketList queue;       // video/audio packets buffered while waiting_thumb

    uint8_t *thumb_data;
    int      thumb_size;

    // Growable per-frame index (item 4: written for real in write_trailer).
    MikuIndexEntry *index;
    int index_len, index_cap;

    // Dedup state (item 3): bytes of the last physically-written, non-duplicate
    // frame, so each new frame only needs a size check + memcmp against this.
    uint8_t *last_frame_data;
    int      last_frame_size;
    int      last_frame_cap;
    int64_t  last_frame_offset;

    // Audio (item 1/6): built up in a dynamic buffer as FLAC packets arrive,
    // written to the real file as one contiguous blob in write_trailer, since
    // the container format wants audio as a single indexed region rather than
    // per-packet entries like video.
    AVIOContext *audio_dyn;
    int64_t      audio_total_samples;

    int64_t video_data_offset;

    // item 2 (VFR) fix: = fps_den/fps_num exactly as written to the header
    // by miku_flush_header_and_write. pkt->duration arrives at
    // miku_write_video_packet already in the *stream's* time_base (that's
    // FFmpeg's packet contract) -- but this container's declared timebase
    // is fps_den/fps_num, chosen from avg_frame_rate and independent of
    // st->time_base. Nothing guarantees those two are the same scale (a
    // finer st->time_base, common for genuinely VFR sources, makes them
    // very much not the same scale), so every duration written to the
    // index has to be explicitly rescaled into this, or the demuxer -- which
    // has no way to know anything other than "durations are already in
    // fps_den/fps_num units" -- reconstructs every later frame's pts wrong.
    AVRational hdr_time_base;
} MikuMuxContext;

// ---- small helpers -------------------------------------------------------

static int miku_index_append(MikuMuxContext *c, int64_t offset,
                              uint32_t size, uint32_t duration)
{
    if (c->index_len == c->index_cap) {
        int new_cap = c->index_cap ? c->index_cap * 2 : 1024;
        MikuIndexEntry *ne = av_realloc_array(c->index, new_cap, sizeof(*ne));
        if (!ne)
            return AVERROR(ENOMEM);
        c->index = ne;
        c->index_cap = new_cap;
    }
    c->index[c->index_len].offset   = offset;
    c->index[c->index_len].size     = size;
    c->index[c->index_len].duration = duration;
    c->index_len++;
    return 0;
}

static void miku_write_json_string(AVIOContext *pb, const char *s)
{
    avio_w8(pb, '"');
    for (; *s; s++) {
        unsigned char ch = *s;
        switch (ch) {
        case '"':  avio_write(pb, (const uint8_t *)"\\\"", 2); break;
        case '\\': avio_write(pb, (const uint8_t *)"\\\\", 2); break;
        case '\n': avio_write(pb, (const uint8_t *)"\\n",  2); break;
        case '\r': avio_write(pb, (const uint8_t *)"\\r",  2); break;
        case '\t': avio_write(pb, (const uint8_t *)"\\t",  2); break;
        default:
            if (ch < 0x20) {
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", ch);
                avio_write(pb, (const uint8_t *)buf, 6);
            } else {
                avio_w8(pb, ch);
            }
        }
    }
    avio_w8(pb, '"');
}

// Serializes s->metadata (the standard AVDictionary every muxer already
// exposes -- item 7: this is the "metadata dict" route, chosen over a custom
// AVOptions/private-class scheme because callers already know how to set
// this via "-metadata key=value" with zero format-specific code on their end)
// into a minimal flat JSON object. Returns a buffer the caller must av_free().
static int miku_build_metadata_json(AVFormatContext *s, uint8_t **out_buf, int *out_size)
{
    AVIOContext *dyn;
    const AVDictionaryEntry *e = NULL;
    int first = 1;
    int ret = avio_open_dyn_buf(&dyn);
    if (ret < 0)
        return ret;

    avio_w8(dyn, '{');
    while ((e = av_dict_iterate(s->metadata, e))) {
        if (!first)
            avio_w8(dyn, ',');
        first = 0;
        miku_write_json_string(dyn, e->key);
        avio_w8(dyn, ':');
        miku_write_json_string(dyn, e->value);
    }
    avio_w8(dyn, '}');

    *out_size = avio_close_dyn_buf(dyn, out_buf);
    return 0;
}

// Writes the 64-byte header (with frame_count/index_offset as 0 placeholders
// -- item 4) and the META block, using whatever thumbnail is currently
// resolved (none, if no attached-pic stream was ever declared).
static int miku_flush_header_and_write(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    AVStream *vst = s->streams[c->video_stream_idx];
    AVStream *ast = s->streams[c->audio_stream_idx];
    uint8_t header[MIKU_HEADER_SIZE];
    uint8_t *json = NULL;
    int json_size = 0;
    int bit_depth;
    AVRational fr;
    int64_t video_offset;
    uint8_t meta_flags = 0;
    int ret;
    // HDR/color metadata (bug report follow-up): captured from the video
    // stream's codecpar, NOT from packet side data. color_primaries/trc/
    // space/range are plain codecpar fields set regardless of what the
    // chosen codec's bitstream can represent; mastering-display and
    // content-light-level are stream-level side data on codecpar itself
    // (coded_side_data), populated generically by libavcodec's encode path
    // for any encoder when the input frames carried it -- this is exactly
    // the mechanism libavformat/movenc.c uses for mdcv/clli boxes, and it
    // works the same way regardless of codec, since it never depends on the
    // codec's own bitstream capability.
    const AVPacketSideData *mdcv_sd, *clli_sd;
    int has_color_info;

    ret = miku_build_metadata_json(s, &json, &json_size);
    if (ret < 0)
        return ret;

    mdcv_sd = av_packet_side_data_get(vst->codecpar->coded_side_data,
                                       vst->codecpar->nb_coded_side_data,
                                       AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
    clli_sd = av_packet_side_data_get(vst->codecpar->coded_side_data,
                                       vst->codecpar->nb_coded_side_data,
                                       AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
    has_color_info = vst->codecpar->color_primaries != AVCOL_PRI_UNSPECIFIED ||
                      vst->codecpar->color_trc       != AVCOL_TRC_UNSPECIFIED ||
                      vst->codecpar->color_space      != AVCOL_SPC_UNSPECIFIED ||
                      vst->codecpar->color_range      != AVCOL_RANGE_UNSPECIFIED ||
                      mdcv_sd || clli_sd;
    // Deliberately all-or-nothing: if nothing was ever set, we write nothing
    // and the demuxer leaves codecpar at its own UNSPECIFIED defaults, so
    // SDR sources never get false color tags injected.

    // Nominal FPS for the header. VFR content (item 2) doesn't need this to be
    // exact -- true per-frame timing lives in the index table's duration
    // field -- but avg_frame_rate (falling back to 1/time_base) gives readers
    // a sane reference rate without decoding the whole index first.
    fr = vst->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0)
        fr = av_inv_q(vst->time_base);
    if (fr.num <= 0 || fr.den <= 0)
        fr = (AVRational){ 30, 1 };
    while (fr.num > 65535 || fr.den > 65535) { // keep it representable in u16/u16
        fr.num = (fr.num + 1) / 2;
        fr.den = (fr.den + 1) / 2;
    }
    // item 2 (VFR) fix: this is the timebase every duration written to the
    // index must end up expressed in -- see hdr_time_base's own comment.
    c->hdr_time_base = (AVRational){ fr.den, fr.num };

    bit_depth = ast->codecpar->bits_per_raw_sample > 0
                    ? ast->codecpar->bits_per_raw_sample : 16;

    memset(header, 0, sizeof(header));
    memcpy(header, "MIKU", 4);
    AV_WL16(header + MIKU_OFF_VERSION,
            c->delta_mode ? MIKU_FORMAT_VERSION_DELTA : MIKU_FORMAT_VERSION);
    AV_WL16(header + MIKU_OFF_FPS_NUM,     fr.num);
    AV_WL16(header + MIKU_OFF_FPS_DEN,     fr.den);
    AV_WL32(header + MIKU_OFF_FRAME_COUNT, 0); // placeholder; patched in write_trailer
    AV_WL32(header + MIKU_OFF_SAMPLE_RATE, ast->codecpar->sample_rate);
    AV_WL16(header + MIKU_OFF_CHANNELS,    ast->codecpar->ch_layout.nb_channels);
    AV_WL16(header + MIKU_OFF_BIT_DEPTH,   bit_depth);
    AV_WL64(header + MIKU_OFF_METADATA_OFF, MIKU_HEADER_SIZE);
    AV_WL64(header + MIKU_OFF_INDEX_OFF,   0); // placeholder; patched in write_trailer
    header[MIKU_OFF_CODEC_FLAG] = c->delta_mode ? MIKU_CODEC_WEBP_DELTA : MIKU_CODEC_WEBP;
    // init() already rejected anything other than WEBP/MIKU_DELTA.

    // SAR: default to 1:1 (square pixels) if the caller never set one.
    // Reduced the same way as fps above -- codecpar's AVRational isn't
    // guaranteed to already be in lowest terms, and a value that overflows
    // u16 would otherwise silently truncate into a bogus ratio.
    {
        AVRational sar = vst->codecpar->sample_aspect_ratio;
        if (sar.num <= 0 || sar.den <= 0)
            sar = (AVRational){ 1, 1 };
        while (sar.num > 65535 || sar.den > 65535) {
            sar.num = (sar.num + 1) / 2;
            sar.den = (sar.den + 1) / 2;
        }
        AV_WL16(header + MIKU_OFF_SAR_NUM, sar.num);
        AV_WL16(header + MIKU_OFF_SAR_DEN, sar.den);
    }

    if (c->thumb_data)
        meta_flags |= MIKU_META_FLAG_THUMBNAIL;

    int color_block_size = 0;
    if (has_color_info) {
        meta_flags |= MIKU_META_FLAG_COLOR;
        color_block_size = 4 /* primaries, trc, space, range */
                         + 1 + (mdcv_sd ? 4 + (int)mdcv_sd->size : 0)   /* has_mdcv + [size + data] */
                         + 1 + (clli_sd ? 4 + (int)clli_sd->size : 0); /* has_clli + [size + data] */
    }

    // Layout: header, "META"+flags+json_len+json, [color block], [thumbnail].
    // Color block is fixed-shape and self-describing (own size prefixes), so
    // it goes before the thumbnail -- keeps the thumbnail's own implicit
    // "everything up to video_offset" length rule simple and unchanged.
    video_offset = MIKU_HEADER_SIZE + 4 /* "META" */ + 1 /* flags */ + 4 /* json_len */
                 + json_size + color_block_size + (c->thumb_data ? c->thumb_size : 0);
    AV_WL64(header + MIKU_OFF_VIDEO_OFF, video_offset);

    avio_write(s->pb, header, MIKU_HEADER_SIZE);

    avio_write(s->pb, "META", 4);
    avio_w8(s->pb, meta_flags);
    avio_wl32(s->pb, json_size);
    avio_write(s->pb, json, json_size);

    if (has_color_info) {
        avio_w8(s->pb, vst->codecpar->color_primaries);
        avio_w8(s->pb, vst->codecpar->color_trc);
        avio_w8(s->pb, vst->codecpar->color_space);
        avio_w8(s->pb, vst->codecpar->color_range);
        if (mdcv_sd) {
            avio_w8(s->pb, 1);
            avio_wl32(s->pb, (uint32_t)mdcv_sd->size);
            avio_write(s->pb, mdcv_sd->data, mdcv_sd->size);
        } else {
            avio_w8(s->pb, 0);
        }
        if (clli_sd) {
            avio_w8(s->pb, 1);
            avio_wl32(s->pb, (uint32_t)clli_sd->size);
            avio_write(s->pb, clli_sd->data, clli_sd->size);
        } else {
            avio_w8(s->pb, 0);
        }
    }

    if (c->thumb_data)
        avio_write(s->pb, c->thumb_data, c->thumb_size);
    av_free(json);

    c->video_data_offset = video_offset;
    c->header_written = 1;
    return 0;
}

static int miku_write_video_packet(AVFormatContext *s, AVPacket *pkt);
static int miku_write_audio_packet(AVFormatContext *s, AVPacket *pkt);

// Writes the header/META block if it's still pending, then replays anything
// that was queued while we were waiting on the thumbnail.
static int miku_resolve_header(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    AVPacket *qpkt;
    int ret;

    if (c->header_written)
        return 0;

    ret = miku_flush_header_and_write(s);
    if (ret < 0)
        return ret;

    qpkt = av_packet_alloc();
    if (!qpkt)
        return AVERROR(ENOMEM);
    while (c->queue.head) {
        ret = ff_packet_list_get(&c->queue, qpkt);
        if (ret < 0)
            break;
        if (qpkt->stream_index == c->video_stream_idx)
            ret = miku_write_video_packet(s, qpkt);
        else if (qpkt->stream_index == c->audio_stream_idx)
            ret = miku_write_audio_packet(s, qpkt);
        av_packet_unref(qpkt);
        if (ret < 0)
            break;
    }
    av_packet_free(&qpkt);
    return ret;
}

// ---- FFOutputFormat callbacks --------------------------------------------

static int miku_init(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    int i;

    c->video_stream_idx = -1;
    c->audio_stream_idx = -1;
    c->thumb_stream_idx = -1;

    // write_trailer patches the header and appends the index after the fact
    // (item 4) -- that's only possible on a seekable output.
    if (!s->pb->seekable) {
        av_log(s, AV_LOG_ERROR,
               ".miku requires a seekable output: the header and index are "
               "finalized with a seek-back after all frames are written, so "
               "piping to a non-seekable destination isn't supported.\n");
        return AVERROR(EINVAL);
    }

    for (i = 0; i < s->nb_streams; i++) {
        AVStream *st = s->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (st->disposition & AV_DISPOSITION_ATTACHED_PIC) {
                if (c->thumb_stream_idx >= 0) {
                    av_log(s, AV_LOG_ERROR, "Only one attached picture (thumbnail) is supported.\n");
                    return AVERROR(EINVAL);
                }
                if (st->codecpar->codec_id != AV_CODEC_ID_WEBP) {
                    av_log(s, AV_LOG_ERROR,
                           "Thumbnail must be WebP (got %s); PNG support was removed in v1.\n",
                           avcodec_get_name(st->codecpar->codec_id));
                    return AVERROR(EINVAL);
                }
                c->thumb_stream_idx = i;
            } else {
                if (c->video_stream_idx >= 0) {
                    av_log(s, AV_LOG_ERROR, "Only one primary video stream is supported.\n");
                    return AVERROR(EINVAL);
                }
                if (st->codecpar->codec_id != AV_CODEC_ID_WEBP &&
                    st->codecpar->codec_id != AV_CODEC_ID_MIKU_DELTA) {
                    av_log(s, AV_LOG_ERROR,
                           ".miku video must be lossless WebP or .miku delta-frame "
                           "(got %s); PNG support was removed.\n",
                           avcodec_get_name(st->codecpar->codec_id));
                    return AVERROR(EINVAL);
                }
                c->video_stream_idx = i;
                c->delta_mode = (st->codecpar->codec_id == AV_CODEC_ID_MIKU_DELTA);
            }
        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (c->audio_stream_idx >= 0 || st->codecpar->codec_id != AV_CODEC_ID_FLAC) {
                av_log(s, AV_LOG_ERROR, "Exactly one FLAC audio stream is required.\n");
                return AVERROR(EINVAL);
            }
            c->audio_stream_idx = i;
        } else {
            av_log(s, AV_LOG_ERROR, "Only video, audio, and an attached picture are supported.\n");
            return AVERROR(EINVAL);
        }
    }

    if (c->video_stream_idx < 0 || c->audio_stream_idx < 0) {
        av_log(s, AV_LOG_ERROR, ".miku requires exactly one video stream and one FLAC audio stream.\n");
        return AVERROR(EINVAL);
    }

    c->waiting_thumb = c->thumb_stream_idx >= 0;
    return 0;
}

static int miku_write_header(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    // Common case: no attached-pic stream was declared, so there is nothing
    // to wait for -- write the real header immediately. If a thumbnail
    // stream *was* declared, defer (miku_resolve_header runs from
    // write_packet once that packet, or proof there isn't one, shows up).
    if (!c->waiting_thumb)
        return miku_flush_header_and_write(s);
    return 0;
}

// Item 3 (dedup) + item 4 (index) + item 2 (VFR durations) + item 5
// (sequential-friendly: dedup only ever compares against the *previous*
// frame, an O(1) check per packet, not an O(n) search across all frames).
static int miku_write_video_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuMuxContext *c = s->priv_data;
    AVStream *vst = s->streams[c->video_stream_idx];
    int64_t offset;
    uint32_t size;
    // item 2 (VFR) fix: pkt->duration arrives in vst->time_base (FFmpeg's
    // packet contract), which is not guaranteed to already equal
    // hdr_time_base -- see hdr_time_base's own comment. Rescale explicitly
    // instead of assuming the raw numbers already line up.
    int64_t scaled_duration = pkt->duration > 0
        ? av_rescale_q(pkt->duration, vst->time_base, c->hdr_time_base) : 0;
    uint32_t duration = scaled_duration > 0 ? (uint32_t)scaled_duration : 1;
    int is_dup = c->last_frame_size > 0 && pkt->size > 0 &&
                 c->last_frame_size == pkt->size &&
                 memcmp(c->last_frame_data, pkt->data, pkt->size) == 0;

    if (is_dup) {
        offset = c->last_frame_offset;
        size   = (uint32_t)c->last_frame_size;
    } else {
        if (pkt->size > c->last_frame_cap) {
            uint8_t *nd = av_realloc(c->last_frame_data, pkt->size);
            if (!nd)
                return AVERROR(ENOMEM);
            c->last_frame_data = nd;
            c->last_frame_cap  = pkt->size;
        }
        if (pkt->size > 0)
            memcpy(c->last_frame_data, pkt->data, pkt->size);
        c->last_frame_size = pkt->size;

        offset = avio_tell(s->pb);
        avio_write(s->pb, pkt->data, pkt->size);
        c->last_frame_offset = offset;
        size = (uint32_t)pkt->size;
    }

    {
        // Delta mode (item: muxer keyframe-bit tracking): steal size's MSB
        // for the keyframe flag rather than growing the 16-byte index entry.
        // WebP/delta frame payloads won't remotely approach 2GB, so the top
        // bit of a byte count is safe to repurpose. This only ever happens
        // when delta_mode is set (video codec_id == AV_CODEC_ID_MIKU_DELTA);
        // with delta_mode off, size_field == size exactly, unchanged from
        // today's behavior.
        //
        // The bit comes from *this* packet's own delta-ness, not from
        // whatever frame the bytes were deduped against above -- offset/size
        // say where the bytes live, this says what frame N's own role is.
        //
        // Deliberately NOT pkt->flags & AV_PKT_FLAG_KEY: verified against
        // libavcodec/encode.c that the generic pipeline force-sets that flag
        // on every packet this encoder produces (AV_CODEC_ID_WEBP's codec
        // descriptor is AV_CODEC_PROP_INTRA_ONLY, decided before the
        // encoder's own init() mutates codec_id for delta mode), so it's 1
        // regardless of what the encoder actually decided. Side data isn't
        // touched by that mechanism -- see AV_PKT_DATA_MIKU_IS_DELTA's own
        // comment in packet.h.
        int is_delta = c->delta_mode &&
            av_packet_get_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, NULL) != NULL;
        uint32_t size_field = size;
        if (c->delta_mode && !is_delta)
            size_field |= 0x80000000u;
        return miku_index_append(c, offset, size_field, duration);
    }
}

// Item 1 (accept encoded FLAC packets) + item 6 (single contiguous audio
// blob, matching what mikudec.c's miku_parse_flac_header expects to find).
static int miku_write_audio_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuMuxContext *c = s->priv_data;
    AVCodecParameters *apar = s->streams[c->audio_stream_idx]->codecpar;
    uint8_t *side;
    size_t side_size = 0;
    int ret;

    if (!c->audio_dyn) {
        if (apar->extradata_size != FLAC_STREAMINFO_SIZE) {
            av_log(s, AV_LOG_ERROR,
                   "FLAC extradata is %d bytes, expected %d (STREAMINFO) -- "
                   "was the audio stream actually opened with the flac encoder?\n",
                   apar->extradata_size, FLAC_STREAMINFO_SIZE);
            return AVERROR(EINVAL);
        }
        ret = avio_open_dyn_buf(&c->audio_dyn);
        if (ret < 0)
            return ret;
        // ff_flac_write_header is the exact helper FFmpeg's own raw FLAC
        // muxer (libavformat/flacenc.c) and the Matroska muxer use to wrap
        // STREAMINFO in "fLaC" + a metadata-block header. last_block=1
        // because our container has its own separate META block for JSON
        // metadata, so STREAMINFO can be the only (and thus final) FLAC-
        // native metadata block -- frame data starts immediately after it,
        // which is exactly what mikudec.c's parser already expects.
        ret = ff_flac_write_header(c->audio_dyn, apar->extradata, apar->extradata_size, 1);
        if (ret < 0)
            return ret;
    }

    avio_write(c->audio_dyn, pkt->data, pkt->size);
    if (pkt->duration > 0)
        c->audio_total_samples += pkt->duration;

    // The FLAC encoder can finalize STREAMINFO (accurate total sample count,
    // true min/max frame size) only once it has seen the whole stream, and
    // signals that by attaching the updated 34 bytes as side data on a later
    // packet -- normally the last one. STREAMINFO sits at a fixed offset (8
    // bytes: "fLaC" + the 4-byte metadata-block header) into the blob we're
    // building, so patching it is a seek-back within the dynamic buffer, not
    // a resize. Same pattern libavformat/flacenc.c uses on its own output.
    side = av_packet_get_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA, &side_size);
    if (side && side_size == FLAC_STREAMINFO_SIZE) {
        int64_t save_pos = avio_tell(c->audio_dyn);
        avio_seek(c->audio_dyn, 8, SEEK_SET);
        avio_write(c->audio_dyn, side, FLAC_STREAMINFO_SIZE);
        avio_seek(c->audio_dyn, save_pos, SEEK_SET);
    }

    return 0;
}

static int miku_write_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuMuxContext *c = s->priv_data;
    int ret;

    if (pkt->stream_index == c->thumb_stream_idx) {
        // The thumbnail arrives as one dedicated packet; it never becomes a
        // frame-index entry or part of the audio blob.
        if (c->header_written)
            return 0; // late/duplicate attached-pic packet; ignore
        av_freep(&c->thumb_data);
        c->thumb_data = pkt->size ? av_memdup(pkt->data, pkt->size) : NULL;
        if (pkt->size && !c->thumb_data)
            return AVERROR(ENOMEM);
        c->thumb_size = pkt->size;
        c->waiting_thumb = 0;
        return miku_resolve_header(s);
    }

    if (!c->header_written) {
        if (c->waiting_thumb)
            return ff_packet_list_put(&c->queue, pkt, NULL, 0);
        ret = miku_resolve_header(s); // no thumbnail stream at all; nothing to wait for
        if (ret < 0)
            return ret;
    }

    if (pkt->stream_index == c->video_stream_idx)
        return miku_write_video_packet(s, pkt);
    if (pkt->stream_index == c->audio_stream_idx)
        return miku_write_audio_packet(s, pkt);

    return 0; // stream init() didn't recognize; shouldn't happen
}

static int miku_write_trailer(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    uint8_t *audio_buf = NULL;
    int audio_size = 0;
    int64_t audio_offset, index_offset, end_pos;
    uint32_t audio_samples_u32;
    int i, ret;

    if (!c->header_written) {
        ret = miku_resolve_header(s); // zero packets ever arrived, or the
        if (ret < 0)                  // thumbnail we waited for never came
            return ret;
    }

    if (c->audio_dyn)
        audio_size = avio_close_dyn_buf(c->audio_dyn, &audio_buf);
    c->audio_dyn = NULL;

    audio_offset = avio_tell(s->pb);
    if (audio_buf && audio_size > 0)
        avio_write(s->pb, audio_buf, audio_size);
    av_free(audio_buf);

    // Item 4: the real index, written now that every frame's offset/size and
    // the audio blob's location are finally known.
    index_offset = avio_tell(s->pb);
    for (i = 0; i < c->index_len; i++) {
        avio_wl64(s->pb, c->index[i].offset);
        avio_wl32(s->pb, c->index[i].size);
        avio_wl32(s->pb, c->index[i].duration);
    }
    audio_samples_u32 = c->audio_total_samples > 0xFFFFFFFFLL
                             ? 0xFFFFFFFFU : (uint32_t)c->audio_total_samples;
    avio_wl64(s->pb, audio_offset);
    avio_wl32(s->pb, (uint32_t)audio_size);
    avio_wl32(s->pb, audio_samples_u32); // mikudec.c doesn't read this field today; kept for forward compatibility

    end_pos = avio_tell(s->pb);

    // Seek back and patch the only two header fields that couldn't be known
    // up front (item 4).
    avio_seek(s->pb, MIKU_OFF_FRAME_COUNT, SEEK_SET);
    avio_wl32(s->pb, (uint32_t)c->index_len);
    avio_seek(s->pb, MIKU_OFF_INDEX_OFF, SEEK_SET);
    avio_wl64(s->pb, index_offset);
    avio_seek(s->pb, end_pos, SEEK_SET);

    return 0;
}

static void miku_deinit(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    uint8_t *tmp;

    av_freep(&c->index);
    av_freep(&c->last_frame_data);
    av_freep(&c->thumb_data);
    if (c->audio_dyn) {
        avio_close_dyn_buf(c->audio_dyn, &tmp);
        av_free(tmp);
        c->audio_dyn = NULL;
    }
    ff_packet_list_free(&c->queue);
}

const FFOutputFormat ff_miku_muxer = {
    .p.name         = "miku",
    .p.long_name    = NULL_IF_CONFIG_SMALL("MIKU Lossless Video"),
    .p.extensions   = "miku",
    // v1 is WebP-only (init() rejects anything else); WebP is also the
    // default a caller gets if -c:v isn't passed explicitly.
    .p.audio_codec  = AV_CODEC_ID_FLAC,
    .p.video_codec  = AV_CODEC_ID_WEBP,
    .p.flags        = AVFMT_GLOBALHEADER | AVFMT_VARIABLE_FPS,
    .priv_data_size = sizeof(MikuMuxContext),
    .init           = miku_init,
    .write_header   = miku_write_header,
    .write_packet   = miku_write_packet,
    .write_trailer  = miku_write_trailer,
    .deinit         = miku_deinit,
};
