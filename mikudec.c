// mikudec.c — MIKU Lossless Video demuxer for FFmpeg
//
// Container: magic "MIKU", 64-byte header, per-frame index table (16 bytes/entry:
// offset u64, size u32, duration u32), one trailing index entry for the audio blob,
// video payload = lossless WebP frames, audio payload = a complete embedded FLAC
// stream. Codec flag 0x01 (WebP) or 0x02 (WebP-delta, opt-in) are the only accepted
// values (0x00, PNG, was supported pre-v1 and has been removed -- see miku_read_header).
// In delta mode only, each video index entry's size field has its MSB stolen as a
// keyframe flag; see miku_get_index_entry and MIKU_CODEC_WEBP_DELTA in mikumux.c for
// the full picture. Off by default -- non-delta files are read exactly as before.

#include "avformat.h"
#include "demux.h"
#include "internal.h"
#include "libavcodec/flac.h"
#include "libavcodec/defs.h"
#include "libavcodec/packet.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "libavutil/avstring.h"
#include "libavutil/mastering_display_metadata.h"

#define MIKU_MAGIC             MKTAG('M','I','K','U')
#define MIKU_HEADER_SIZE       64
#define MIKU_INDEX_ENTRY_SIZE  16
#define MIKU_CODEC_FLAG_OFFSET 45
#define MIKU_OFF_METADATA_OFF  22
#define MIKU_OFF_SAR_NUM       54
#define MIKU_OFF_SAR_DEN       56
#define MIKU_META_FLAG_THUMBNAIL 0x02
#define MIKU_META_FLAG_COLOR     0x04
#define MIKU_CODEC_WEBP        0x01
#define MIKU_CODEC_WEBP_DELTA  0x02  // see mikumux.c -- keyframe bit stolen from each
                                      // video index entry's size MSB when this is set

// Above this many frames, don't preload the whole index into RAM (item 7).
// 1,000,000 frames in memory is ~16MB (3 arrays x 4-8 bytes); comfortably
// safe, but a file with tens of millions of frames could still add up, so
// we fall back to reading index entries from disk on demand past this point.
#define MIKU_INDEX_PRELOAD_MAX 1000000

// Minimum audio read size (item 6). Also used as the interleaving grain --
// see miku_read_packet().
#define MIKU_AUDIO_CHUNK_SIZE  65536

typedef struct MikuContext {
    // ---- Video frame index ----
    int64_t *frame_offsets;      // NULL when index_in_memory == 0
    int32_t *frame_sizes;        // always the real byte count -- MSB (if any) is
                                  // masked off before storage, see miku_read_header
    int32_t *frame_durations;
    uint8_t *frame_keyframe;     // 1 byte/frame, NULL unless delta_mode (index_in_memory
                                  // path only -- the streaming/on-demand path re-derives
                                  // it per read in miku_get_index_entry instead)
    int      num_frames;
    int      current_frame;
    int64_t  next_pts;           // item 2 (VFR) fix: running sum of real per-frame
                                  // durations. Frame i's pts is the sum of frames
                                  // [0, i)'s durations, NOT i itself -- durations
                                  // vary for real (this format is AVFMT_VARIABLE_FPS
                                  // and mikumux.c's own header comment says true
                                  // per-frame timing lives in this field), so a
                                  // demuxer that assumes 1 tick = 1 frame silently
                                  // mistimes every frame after the first one whose
                                  // duration isn't exactly 1. See
                                  // miku_read_video_packet and miku_frame_for_pts.

    int      delta_mode;         // 1 iff header codec_flag == MIKU_CODEC_WEBP_DELTA;
                                  // gates keyframe-bit extraction from the index. When 0,
                                  // every frame is implicitly a keyframe, exactly as today.
    int      index_in_memory;    // 1 = arrays above are valid; 0 = streaming mode
    int64_t  index_table_offset; // file offset of the index table (streaming mode)

    // ---- Audio (FLAC) ----
    int64_t audio_frame_start;   // first byte of real FLAC frame data
    int64_t audio_frame_end;     // end of the audio blob (audio_offset + audio_size)
    int64_t audio_cursor;        // next unread byte in [audio_frame_start, audio_frame_end)

    // ---- Interleaving ----
    int video_frames_per_audio_chunk; // 0 = no audio to interleave
    int video_frames_since_audio;

    // ---- Shared I/O position cache ----
    // Tracks where we believe the underlying file descriptor currently sits,
    // so miku_read_video_packet/miku_read_audio_packet can skip a redundant
    // avio_seek() when the previous read already left it in the right place
    // (item 5). -1 means "unknown, must seek before the next read."
    int64_t file_pos;

    // ---- Container header fields ----
    int fps_num, fps_den;
    int sample_rate;
    int channels;
    int bits_per_sample;
} MikuContext;

static int miku_probe(const AVProbeData *p)
{
    if (p->buf_size < 4) return 0;
    if (AV_RL32(p->buf) == MIKU_MAGIC) return AVPROBE_SCORE_MAX;
    return 0;
}

// Parses the embedded FLAC file's own "fLaC" magic + metadata block chain
// (starting at audio_offset) to locate the real STREAMINFO block and the
// byte offset where actual FLAC *frame* data begins. STREAMINFO is copied
// verbatim into codecpar->extradata -- this mirrors exactly what FFmpeg's
// own libavformat/flacdec.c does for its extradata, down to the 34-byte
// FLAC_STREAMINFO_SIZE (verified against current FFmpeg source).
static int miku_parse_flac_header(MikuContext *m, AVIOContext *pb,
                                   AVStream *ast, int64_t audio_offset,
                                   int64_t audio_blob_end)
{
    uint8_t magic[4];
    uint8_t block_header[4];
    int ret, is_last, block_type, block_size;
    int64_t pos = audio_offset;

    avio_seek(pb, pos, SEEK_SET);
    ret = avio_read(pb, magic, 4);
    if (ret != 4 || AV_RL32(magic) != MKTAG('f','L','a','C'))
        return AVERROR_INVALIDDATA;
    pos += 4;

    for (;;) {
        if (pos + 4 > audio_blob_end)
            return AVERROR_INVALIDDATA;

        ret = avio_read(pb, block_header, 4);
        if (ret != 4)
            return AVERROR_INVALIDDATA;

        flac_parse_block_header(block_header, &is_last, &block_type, &block_size);

        if (pos + 4 + (int64_t)block_size > audio_blob_end)
            return AVERROR_INVALIDDATA;

        if (block_type == FLAC_METADATA_TYPE_STREAMINFO && !ast->codecpar->extradata) {
            if (block_size != FLAC_STREAMINFO_SIZE)
                return AVERROR_INVALIDDATA;
            ast->codecpar->extradata = av_mallocz(FLAC_STREAMINFO_SIZE + AV_INPUT_BUFFER_PADDING_SIZE);
            if (!ast->codecpar->extradata)
                return AVERROR(ENOMEM);
            ast->codecpar->extradata_size = FLAC_STREAMINFO_SIZE;
            ret = avio_read(pb, ast->codecpar->extradata, FLAC_STREAMINFO_SIZE);
            if (ret != FLAC_STREAMINFO_SIZE)
                return AVERROR_INVALIDDATA;
        } else {
            avio_skip(pb, block_size);
        }

        pos += 4 + block_size;
        if (is_last)
            break;
    }

    if (!ast->codecpar->extradata)
        return AVERROR_INVALIDDATA; // STREAMINFO is mandatory in a valid FLAC stream

    m->audio_frame_start = pos;
    return 0;
}

// Reads the META block at metadata_offset and, if present, reconstructs
// color/HDR metadata onto vst->codecpar (bug-report follow-up: WebP's
// bitstream has no field for this, so it's captured/restored at the
// container level instead, mirroring how libavformat/movenc.c and mov.c
// handle mdcv/clli). Only the color sub-block is actually parsed here; the
// JSON metadata and any thumbnail are skipped over (not yet exposed by this
// demuxer).
static int miku_parse_meta_block(AVFormatContext *s, AVStream *vst, int64_t metadata_offset)
{
    AVIOContext *pb = s->pb;
    uint8_t magic[4];
    uint8_t flags;
    uint32_t json_len;
    int ret;

    avio_seek(pb, metadata_offset, SEEK_SET);
    ret = avio_read(pb, magic, 4);
    if (ret != 4 || memcmp(magic, "META", 4) != 0)
        return 0; // no META block (older file, or offset unset) -- not an error
    flags = avio_r8(pb);
    json_len = avio_rl32(pb);
    avio_skip(pb, json_len);

    if (!(flags & MIKU_META_FLAG_COLOR))
        return 0;

    vst->codecpar->color_primaries = avio_r8(pb);
    vst->codecpar->color_trc       = avio_r8(pb);
    vst->codecpar->color_space     = avio_r8(pb);
    vst->codecpar->color_range     = avio_r8(pb);

    if (avio_r8(pb)) { // has_mastering_display
        uint32_t sz = avio_rl32(pb);
        if (sz == sizeof(AVMasteringDisplayMetadata)) {
            AVPacketSideData *sd = av_packet_side_data_new(
                &vst->codecpar->coded_side_data, &vst->codecpar->nb_coded_side_data,
                AV_PKT_DATA_MASTERING_DISPLAY_METADATA, sz, 0);
            if (!sd)
                return AVERROR(ENOMEM);
            ret = avio_read(pb, sd->data, sz);
            if (ret != (int)sz)
                return AVERROR_INVALIDDATA;
        } else {
            // Size doesn't match this build's struct layout (e.g. a file
            // written by a different FFmpeg version) -- skip rather than
            // reinterpret bytes that wouldn't mean what we think.
            avio_skip(pb, sz);
        }
    }

    if (avio_r8(pb)) { // has_content_light
        uint32_t sz = avio_rl32(pb);
        if (sz == sizeof(AVContentLightMetadata)) {
            AVPacketSideData *sd = av_packet_side_data_new(
                &vst->codecpar->coded_side_data, &vst->codecpar->nb_coded_side_data,
                AV_PKT_DATA_CONTENT_LIGHT_LEVEL, sz, 0);
            if (!sd)
                return AVERROR(ENOMEM);
            ret = avio_read(pb, sd->data, sz);
            if (ret != (int)sz)
                return AVERROR_INVALIDDATA;
        } else {
            avio_skip(pb, sz);
        }
    }

    return 0;
}

static int miku_read_header(AVFormatContext *s)
{
    MikuContext *m = s->priv_data;
    AVIOContext *pb = s->pb;
    AVStream *vst, *ast;
    uint8_t header[MIKU_HEADER_SIZE];
    uint8_t codec_flag;
    int64_t idx_off, audio_idx_pos, audio_offset;
    int32_t audio_size;
    uint8_t entry[MIKU_INDEX_ENTRY_SIZE];
    int ret;

    ret = avio_read(pb, header, MIKU_HEADER_SIZE);
    if (ret != MIKU_HEADER_SIZE) return AVERROR_INVALIDDATA;
    if (AV_RL32(header) != MIKU_MAGIC) return AVERROR_INVALIDDATA;

    m->fps_num = AV_RL16(header + 6);
    m->fps_den = AV_RL16(header + 8);
    if (m->fps_den == 0) m->fps_den = 1;
    m->num_frames       = AV_RL32(header + 10);
    m->sample_rate      = AV_RL32(header + 14);
    m->channels         = AV_RL16(header + 18);
    m->bits_per_sample  = AV_RL16(header + 20);
    idx_off             = AV_RL64(header + 30);
    codec_flag          = header[MIKU_CODEC_FLAG_OFFSET];

    if (m->num_frames <= 0) return AVERROR_INVALIDDATA;

    // ---- Video stream ----
    vst = avformat_new_stream(s, NULL);
    if (!vst) return AVERROR(ENOMEM);
    vst->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

    // WebP (v1) or WebP-delta (opt-in delta mode) are the only accepted
    // values. Earlier format versions also supported PNG (codec flag 0x00);
    // that path produced scrambled video when a file's actual codec didn't
    // match the hardcoded decoder, and has been removed rather than fixed.
    if (codec_flag != MIKU_CODEC_WEBP && codec_flag != MIKU_CODEC_WEBP_DELTA) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported .miku codec flag 0x%02x (PNG support was removed; "
               "only WebP (0x01) or WebP-delta (0x02) are accepted).\n", codec_flag);
        return AVERROR_INVALIDDATA;
    }
    m->delta_mode = (codec_flag == MIKU_CODEC_WEBP_DELTA);
    vst->codecpar->codec_id = m->delta_mode ? AV_CODEC_ID_MIKU_DELTA : AV_CODEC_ID_WEBP;
    // codecpar->format is deliberately left at its default (AV_PIX_FMT_NONE):
    // WebP is a self-describing compressed bitstream, so the decoder
    // determines the real pixel format from each frame's own header at
    // decode time -- nothing here needs to (or should) declare it upfront.

    ret = miku_parse_meta_block(s, vst, AV_RL64(header + MIKU_OFF_METADATA_OFF));
    if (ret < 0) return ret;

    // SAR: 1:1 (square pixels) unless the header says otherwise. A 0 in
    // either field means "not set" -- older/foreign files may not have
    // written anything meaningful here, so treat that as unknown rather
    // than dividing by zero or forcing a specific ratio.
    {
        int sar_num = AV_RL16(header + MIKU_OFF_SAR_NUM);
        int sar_den = AV_RL16(header + MIKU_OFF_SAR_DEN);
        if (sar_num > 0 && sar_den > 0) {
            vst->codecpar->sample_aspect_ratio.num = sar_num;
            vst->codecpar->sample_aspect_ratio.den = sar_den;
        }
    }

    avpriv_set_pts_info(vst, 64, m->fps_den, m->fps_num);
    vst->duration  = m->num_frames;
    vst->nb_frames = m->num_frames;

    // ---- Index table (item 4 groundwork + item 7 streaming fallback) ----
    if (m->num_frames <= MIKU_INDEX_PRELOAD_MAX) {
        m->index_in_memory = 1;
        m->frame_offsets   = av_calloc(m->num_frames, sizeof(int64_t));
        m->frame_sizes     = av_calloc(m->num_frames, sizeof(int32_t));
        m->frame_durations = av_calloc(m->num_frames, sizeof(int32_t));
        if (!m->frame_offsets || !m->frame_sizes || !m->frame_durations)
            return AVERROR(ENOMEM);
        if (m->delta_mode) {
            m->frame_keyframe = av_calloc(m->num_frames, sizeof(uint8_t));
            if (!m->frame_keyframe)
                return AVERROR(ENOMEM);
        }

        avio_seek(pb, idx_off, SEEK_SET);
        for (int i = 0; i < m->num_frames; i++) {
            uint32_t raw_size;
            ret = avio_read(pb, entry, MIKU_INDEX_ENTRY_SIZE);
            if (ret != MIKU_INDEX_ENTRY_SIZE) return AVERROR_INVALIDDATA;
            m->frame_offsets[i]   = AV_RL64(entry);
            raw_size              = AV_RL32(entry + 8);
            m->frame_durations[i] = AV_RL32(entry + 12);
            if (m->delta_mode) {
                m->frame_keyframe[i] = (raw_size & 0x80000000u) ? 1 : 0;
                m->frame_sizes[i]    = (int32_t)(raw_size & 0x7FFFFFFFu);
            } else {
                m->frame_sizes[i]    = (int32_t)raw_size;
            }
        }
    } else {
        // Streaming-index fallback: read each entry from disk on demand in
        // miku_get_index_entry() instead of allocating O(num_frames) memory.
        m->index_in_memory    = 0;
        m->index_table_offset = idx_off;
    }

    // ---- Trailing index entry: audio blob location/size ----
    audio_idx_pos = idx_off + (int64_t)m->num_frames * MIKU_INDEX_ENTRY_SIZE;
    avio_seek(pb, audio_idx_pos, SEEK_SET);
    ret = avio_read(pb, entry, MIKU_INDEX_ENTRY_SIZE);
    if (ret != MIKU_INDEX_ENTRY_SIZE) return AVERROR_INVALIDDATA;
    audio_offset = AV_RL64(entry);
    audio_size   = AV_RL32(entry + 8);

    // ---- Audio stream (FLAC) ----
    ast = avformat_new_stream(s, NULL);
    if (!ast) return AVERROR(ENOMEM);
    ast->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    ast->codecpar->codec_id   = AV_CODEC_ID_FLAC;
    ast->codecpar->sample_rate = m->sample_rate;
    av_channel_layout_default(&ast->codecpar->ch_layout, m->channels);
    ast->codecpar->bits_per_coded_sample = m->bits_per_sample;
    avpriv_set_pts_info(ast, 64, 1, m->sample_rate);

    // BUG FIX #3 (audio): FLAC is compressed and frame-based, unlike raw PCM.
    // Slicing it at arbitrary byte boundaries (the old fixed 4096-byte chunks
    // / WAV "+44" skip) does not yield decodable frames -- it only ever
    // "worked" for PCM because PCM has no internal framing to break.
    // Fix has two parts:
    //   1. Parse past "fLaC" + metadata blocks to find real frame data and
    //      copy STREAMINFO into extradata (miku_parse_flac_header, mirrors
    //      libavformat/flacdec.c).
    //   2. Mark the stream AVSTREAM_PARSE_FULL_RAW so libavformat's own FLAC
    //      parser re-splits whatever byte chunks we hand it into correctly
    //      bounded, correctly timestamped frames -- verified against current
    //      FFmpeg source (libavformat/flacdec.c uses the identical pattern).
    ret = miku_parse_flac_header(m, pb, ast, audio_offset, audio_offset + (int64_t)audio_size);
    if (ret < 0) return ret;
    m->audio_frame_end = audio_offset + audio_size;
    ffstream(ast)->need_parsing = AVSTREAM_PARSE_FULL_RAW;

    // ---- Interleaving plan ----
    // The old code served every video packet before any audio packet, which
    // works but risks starving ffplay's audio queue on a large file (it will
    // fill its bounded video-packet queue and stall long before demuxing
    // reaches the first audio byte). Instead we spread MIKU_AUDIO_CHUNK_SIZE
    // audio reads evenly across the video frame sequence -- see
    // miku_read_packet().
    {
        int64_t audio_bytes = m->audio_frame_end - m->audio_frame_start;
        int64_t est_chunks = (audio_bytes + MIKU_AUDIO_CHUNK_SIZE - 1) / MIKU_AUDIO_CHUNK_SIZE;
        if (est_chunks > 0) {
            int per = (int)(m->num_frames / est_chunks);
            m->video_frames_per_audio_chunk = per > 0 ? per : 1;
        } else {
            m->video_frames_per_audio_chunk = 0;
        }
    }

    m->current_frame  = 0;
    m->next_pts        = 0;
    m->audio_cursor    = m->audio_frame_start;
    m->file_pos         = -1; // force a seek before the very first read_packet
    return 0;
}

// Fetches index entry i regardless of whether the table is fully in memory
// or being streamed from disk on demand (item 7). is_keyframe is always 1
// when delta_mode is off, matching today's behavior (every frame stands
// alone). When delta_mode is on, it reflects the MSB stolen from size --
// masked out here before *size is ever handed to a caller, so the existing
// "size < 0 is corrupt data" guard downstream never sees a raw stolen bit.
static int miku_get_index_entry(AVFormatContext *s, int i,
                                 int64_t *offset, int32_t *size, int32_t *duration,
                                 int *is_keyframe)
{
    MikuContext *m = s->priv_data;

    if (m->index_in_memory) {
        *offset      = m->frame_offsets[i];
        *size        = m->frame_sizes[i]; // already masked at load time
        *duration    = m->frame_durations[i];
        *is_keyframe = m->delta_mode ? m->frame_keyframe[i] : 1;
        return 0;
    }

    {
        AVIOContext *pb = s->pb;
        uint8_t entry[MIKU_INDEX_ENTRY_SIZE];
        int64_t pos = m->index_table_offset + (int64_t)i * MIKU_INDEX_ENTRY_SIZE;
        uint32_t raw_size;
        int ret;

        avio_seek(pb, pos, SEEK_SET);
        ret = avio_read(pb, entry, MIKU_INDEX_ENTRY_SIZE);
        if (ret != MIKU_INDEX_ENTRY_SIZE) return AVERROR_INVALIDDATA;
        *offset  = AV_RL64(entry);
        raw_size = AV_RL32(entry + 8);
        *duration = AV_RL32(entry + 12);
        if (m->delta_mode) {
            *is_keyframe = (raw_size & 0x80000000u) ? 1 : 0;
            *size        = (int32_t)(raw_size & 0x7FFFFFFFu);
        } else {
            *is_keyframe = 1;
            *size        = (int32_t)raw_size;
        }
        m->file_pos = pos + MIKU_INDEX_ENTRY_SIZE; // fd moved; keep the cache honest
        return 0;
    }
}

static int miku_read_video_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuContext *m = s->priv_data;
    AVIOContext *pb = s->pb;
    int64_t offset;
    int32_t size, duration;
    int is_keyframe;
    int ret;

    ret = miku_get_index_entry(s, m->current_frame, &offset, &size, &duration, &is_keyframe);
    if (ret < 0) return ret;
    if (size < 0 || offset < 0)
        return AVERROR_INVALIDDATA; // corrupt index entry; don't hand a bogus size to av_new_packet

    // OPTIMIZATION (item 5): only seek if the fd isn't already sitting here.
    if (m->file_pos != offset)
        avio_seek(pb, offset, SEEK_SET);

    ret = av_new_packet(pkt, size);
    if (ret < 0) return ret;
    ret = avio_read(pb, pkt->data, pkt->size);
    if (ret != pkt->size) {
        av_packet_unref(pkt);
        return AVERROR_EOF;
    }

    pkt->stream_index = 0;
    pkt->duration     = duration > 0 ? duration : 1;
    pkt->pts          = m->next_pts; // item 2 (VFR): accumulated real duration,
                                      // not frame index -- see next_pts's own comment
    m->next_pts      += pkt->duration;
    // Non-delta files: every frame is an independent lossless image, exactly
    // as before. Delta files: reflects the bit the muxer actually wrote for
    // this frame -- the new stateful decoder (and any generic seek-refinement
    // logic) needs this to be accurate, not just always-true.
    if (is_keyframe)
        pkt->flags |= AV_PKT_FLAG_KEY;

    m->file_pos = offset + size;
    m->current_frame++;
    m->video_frames_since_audio++;
    return 0;
}

static int miku_read_audio_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuContext *m = s->priv_data;
    AVIOContext *pb = s->pb;
    int64_t remaining = m->audio_frame_end - m->audio_cursor;
    int chunk = remaining < MIKU_AUDIO_CHUNK_SIZE ? (int)remaining : MIKU_AUDIO_CHUNK_SIZE;
    int ret;

    if (chunk <= 0) return AVERROR_EOF;

    // OPTIMIZATION (item 5, same cache as video): skip the seek if sequential.
    if (m->file_pos != m->audio_cursor)
        avio_seek(pb, m->audio_cursor, SEEK_SET);

    // OPTIMIZATION (item 6): 65536-byte reads instead of 4096. We deliberately
    // do NOT read the whole FLAC blob into one packet -- see the chat reply
    // for why that would undo the interleaving above and reintroduce the
    // queue-starvation problem for large files.
    ret = av_new_packet(pkt, chunk);
    if (ret < 0) return ret;
    ret = avio_read(pb, pkt->data, pkt->size);
    if (ret != pkt->size) {
        av_packet_unref(pkt);
        return AVERROR_EOF;
    }

    pkt->stream_index = 1;
    pkt->pts = AV_NOPTS_VALUE; // the FLAC parser (AVSTREAM_PARSE_FULL_RAW) derives real timestamps

    m->audio_cursor += chunk;
    m->file_pos       = m->audio_cursor;
    m->video_frames_since_audio = 0;
    return 0;
}

static int miku_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuContext *m = s->priv_data;
    int video_left = m->current_frame < m->num_frames;
    int audio_left = m->audio_cursor < m->audio_frame_end;

    if (!video_left && !audio_left)
        return AVERROR_EOF;

    if (video_left && audio_left) {
        int due_for_audio = m->video_frames_per_audio_chunk > 0 &&
                             m->video_frames_since_audio >= m->video_frames_per_audio_chunk;
        return due_for_audio ? miku_read_audio_packet(s, pkt)
                              : miku_read_video_packet(s, pkt);
    }

    return video_left ? miku_read_video_packet(s, pkt)
                       : miku_read_audio_packet(s, pkt);
}

// item 2 (VFR) fix: finds the largest frame index i whose accumulated pts
// (sum of frames [0, i)'s durations) is <= target_pts -- i.e. the frame
// that's actually current at that point in time. Needed because pts is no
// longer simply the frame index (see next_pts's own comment); works
// unmodified for both index_in_memory and streaming mode since
// miku_get_index_entry already abstracts that difference. Cost is O(target
// frame) index-entry reads -- for index_in_memory that's a fast in-RAM scan
// (bounded by MIKU_INDEX_PRELOAD_MAX); for streaming mode it's genuinely
// O(n) disk reads from the start of the file, the honest price of a correct
// seek when durations aren't uniform and there's no precomputed cumulative
// table to binary-search instead.
static int64_t miku_frame_for_pts(AVFormatContext *s, int64_t target_pts)
{
    MikuContext *m = s->priv_data;
    int64_t cum = 0;
    int i;

    for (i = 0; i < m->num_frames - 1; i++) {
        int64_t off; int32_t sz, dur; int is_kf;
        int64_t d;
        if (miku_get_index_entry(s, i, &off, &sz, &dur, &is_kf) < 0)
            break; // best effort -- caller clamps into [0, num_frames)
        d = dur > 0 ? dur : 1;
        if (cum + d > target_pts)
            break;
        cum += d;
    }
    return i;
}

// item 2 (VFR) fix, companion to the above: sum of frames [0, upto_idx)'s
// durations -- the pts frame upto_idx would receive from the normal
// sequential accumulation in miku_read_video_packet. miku_read_seek needs
// this to re-sync next_pts with wherever current_frame actually ends up,
// including after the delta-mode keyframe snap below, which can land on an
// earlier frame than the seek target itself resolved to.
static int64_t miku_cumulative_duration(AVFormatContext *s, int64_t upto_idx)
{
    MikuContext *m = s->priv_data;
    int64_t cum = 0;
    int i;

    for (i = 0; i < upto_idx && i < m->num_frames; i++) {
        int64_t off; int32_t sz, dur; int is_kf;
        if (miku_get_index_entry(s, i, &off, &sz, &dur, &is_kf) < 0)
            break; // best effort -- a corrupt tail doesn't need to fail the seek
        cum += dur > 0 ? dur : 1;
    }
    return cum;
}

// BUG FIX #4: frame-accurate seeking via the index table (item 2 fix: no
// longer O(1) in general -- see miku_frame_for_pts -- since frame_idx and
// pts stopped being interchangeable the moment duration stopped being
// uniformly 1).
static int miku_read_seek(AVFormatContext *s, int stream_index,
                           int64_t timestamp, int flags)
{
    MikuContext *m = s->priv_data;
    int64_t frame_idx;

    if (m->num_frames <= 0)
        return AVERROR(ENOSYS);

    if (stream_index == 0) {
        frame_idx = miku_frame_for_pts(s, timestamp);
    } else if (stream_index == 1) {
        if (m->sample_rate <= 0 || m->fps_num <= 0)
            return AVERROR(ENOSYS);
        // audio time_base is 1/sample_rate; convert samples -> seconds -> frame index
        double seconds = (double)timestamp / (double)m->sample_rate;
        frame_idx = (int64_t)(seconds * m->fps_num / m->fps_den);
    } else {
        return AVERROR(ENOSYS);
    }

    if (frame_idx < 0) frame_idx = 0;
    if (frame_idx >= m->num_frames) frame_idx = m->num_frames - 1;

    {
        int64_t video_frame_idx = frame_idx;

        // Delta chains only carry real decoder state forward from a
        // keyframe (see the new decoder) -- decoding must start there, not
        // mid-chain. Walk back to the nearest keyframe at or before
        // frame_idx; normal sequential read_packet (and the decoder) catches
        // up to the originally requested frame from there. Non-delta files
        // skip this entirely: every frame is already a keyframe, so seeking
        // stays exact, unchanged from before delta mode existed.
        if (m->delta_mode) {
            int steps = 0;
            for (;;) {
                int64_t off; int32_t sz, dur; int is_kf;
                int ret = miku_get_index_entry(s, video_frame_idx, &off, &sz, &dur, &is_kf);
                if (ret < 0) return ret;
                if (is_kf) break;
                // Bounded at one block's worth of frames (the spec's 100-frame
                // cap): an unbroken non-keyframe run longer than that is a
                // corrupt index, not a legitimate chain -- fail cleanly rather
                // than walking back through the entire file.
                if (video_frame_idx == 0 || ++steps > 100)
                    return AVERROR_INVALIDDATA;
                video_frame_idx--;
            }
        }
        m->current_frame = video_frame_idx;
        // item 2 (VFR) fix: next_pts is a running counter (see its own
        // comment) that miku_read_video_packet only ever advances forward
        // sequentially -- a seek jumps current_frame directly and must
        // re-sync next_pts to match, or every packet after this seek
        // inherits whatever stale value was left over from before it,
        // regardless of which direction or how far we just jumped.
        m->next_pts = miku_cumulative_duration(s, video_frame_idx);
    }

    // Video seeking snaps to the nearest preceding keyframe above (exact,
    // i.e. a no-op snap, for non-delta files). Audio (compressed FLAC) can
    // only jump to an estimated byte offset proportional to playback
    // position -- the FLAC parser resynchronizes on the next valid frame
    // sync it finds, so audio recovers within a fraction of a second rather
    // than being frame-exact. This deliberately keeps using the original,
    // unsnapped frame_idx: audio has no "decode forward from a keyframe and
    // catch up" mechanism the way video does, so the pre-snap position stays
    // closer to what was actually requested. A byte-exact seek would need a
    // sample-position -> byte-offset table (e.g. a FLAC SEEKTABLE block),
    // which miku-pack doesn't currently write.
    if (m->audio_frame_end > m->audio_frame_start) {
        double fraction = (double)frame_idx / (double)m->num_frames;
        int64_t span = m->audio_frame_end - m->audio_frame_start;
        int64_t target = m->audio_frame_start + (int64_t)(fraction * (double)span);
        if (target < m->audio_frame_start) target = m->audio_frame_start;
        if (target > m->audio_frame_end)   target = m->audio_frame_end;
        m->audio_cursor = target;
    }

    m->video_frames_since_audio = 0;
    m->file_pos = -1; // no longer sequential; force a seek on the next read
    return 0;
}

static int miku_read_close(AVFormatContext *s)
{
    MikuContext *m = s->priv_data;
    av_freep(&m->frame_offsets);
    av_freep(&m->frame_sizes);
    av_freep(&m->frame_durations);
    av_freep(&m->frame_keyframe);
    return 0;
}

const FFInputFormat ff_miku_demuxer = {
    .p.name         = "miku",
    .p.long_name    = NULL_IF_CONFIG_SMALL("MIKU Lossless Video"),
    .p.extensions   = "miku",
    .p.flags        = AVFMT_NOBINSEARCH | AVFMT_NOGENSEARCH | AVFMT_NO_BYTE_SEEK,
    // Confirmed via a real leak (ASan, a header claiming far more frames
    // than the file actually backs up): without this, read_close() is
    // never called when read_header() fails partway through -- e.g. after
    // frame_offsets/frame_sizes/frame_durations/frame_keyframe are already
    // allocated but the index table read then hits EOF early and returns
    // an error. read_close() already handles partial initialization safely
    // (av_freep is a no-op on still-NULL pointers); it just wasn't being
    // called at all in the failure case without this flag.
    .flags_internal = FF_INFMT_FLAG_INIT_CLEANUP,
    .priv_data_size = sizeof(MikuContext),
    .read_probe     = miku_probe,
    .read_header    = miku_read_header,
    .read_packet    = miku_read_packet,
    .read_seek      = miku_read_seek,
    .read_close     = miku_read_close,
};
