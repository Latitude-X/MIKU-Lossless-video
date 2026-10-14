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
//             per frame. DEDUPLICATED GLOBALLY: a packet whose bytes are
//             identical to ANY earlier packet in the file (not just the
//             previous one) is not written again -- its index entry just
//             points at the first copy. See miku_write_video_packet.
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
//             LONG-TERM REFERENCES (format v3, codec_flag 0x03). A delta frame
//             normally reconstructs against the previous frame. With long-term
//             references it may instead reconstruct against ANY keyframe in
//             the file -- earlier or later, since a keyframe is a standalone
//             image and so can always be decoded on demand. Only if at least
//             one frame actually uses this does the muxer write v3: it patches
//             the header version/codec_flag in write_trailer and widens every
//             index entry (including the trailing audio entry) to 24 bytes:
//                 {offset u64, size u32, duration u32, ref i32, reserved u32}
//             ref is the index of the reference keyframe, or -1 for none (a
//             keyframe, or an ordinary delta against the previous frame). A
//             file that never uses a long-term reference is still written as
//             plain v2 (16-byte entries), so older readers keep opening it.
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

#include <inttypes.h>
#include <string.h>

#include "avformat.h"
#include "mux.h"
#include "packet_internal.h"
#include "flacenc.h"
#include "libavcodec/flac.h"
#include "libavcodec/packet.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/dict.h"
#include "libavutil/hash.h"
#include "libavutil/mem.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"

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

#define MIKU_CODEC_WEBP_DELTA_REFS 0x03 // delta mode + at least one long-term reference
                                         // (24-byte index entries); old readers reject
                                         // this value outright instead of misdecoding.
#define MIKU_FORMAT_VERSION_REFS   3

#define MIKU_META_FLAG_THUMBNAIL 0x02
#define MIKU_META_FLAG_COLOR     0x04

#define MIKU_KEYFRAME_BIT        0x80000000u // delta mode: MSB of an index entry's size

// -dedup modes
#define MIKU_DEDUP_OFF     0
#define MIKU_DEDUP_LAST    1 // only against the immediately preceding frame
#define MIKU_DEDUP_GLOBAL  2 // against every earlier frame in the file

typedef struct MikuIndexEntry {
    int64_t  offset;
    uint32_t size;
    uint32_t duration;
    int32_t  ref;       // long-term reference keyframe's index, or -1
} MikuIndexEntry;

// One slot of a content table: a 128-bit content digest plus the exact byte
// size, mapped to a value. Two tables use this shape:
//   dedup_t -- every unique packet written -> file offset of its first copy
//   kf_t    -- every keyframe written       -> index of the first frame holding it
typedef struct MikuDedupSlot {
    uint8_t  digest[16];
    int64_t  value;
    uint32_t size;
    uint32_t used;
} MikuDedupSlot;

typedef struct MikuTable {
    MikuDedupSlot *slots; // open addressing, power-of-two capacity, <= half full
    uint32_t       cap;
    uint32_t       n;
} MikuTable;

// A stream-copied reference whose keyframe had not been written yet when the
// referencing packet arrived (a forward reference). Resolved by content in
// write_trailer, once every keyframe of the output is known.
typedef struct MikuPendingRef {
    int32_t  frame;     // index of the referencing frame
    uint32_t size;      // size of the referenced keyframe's bytes
    uint8_t  digest[16];
} MikuPendingRef;

typedef struct MikuMuxContext {
    const AVClass *class; // must be first: required by the AVOptions machinery
    int video_stream_idx;   // -1 until resolved in init()
    int audio_stream_idx;
    int thumb_stream_idx;   // -1 if no attached-pic stream was provided

    int auto_thumbnail; // -auto_thumbnail option; see miku_write_packet

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

    // Dedup state. Fast path (exact, byte-compared): bytes of the most
    // recent frame, so a frame identical to its predecessor needs only a
    // size check + memcmp -- no hashing at all. This covers runs of static
    // or all-zero-residual frames, the common case.
    uint8_t *last_frame_data;
    int      last_frame_size;
    int      last_frame_cap;
    int64_t  last_frame_offset;

    // Global dedup (-dedup global): content digest of every unique packet
    // written so far -> where its single physical copy lives. Lets a frame
    // that matches ANY earlier frame (a loop, a returning scene, a repeated
    // slate) reuse that copy. The digest is 128-bit murmur3 plus the exact
    // byte size; there is no byte-for-byte verification against the old copy
    // (the muxer cannot read back what it has written), so correctness rests
    // on the digest -- an accidental collision between two distinct packets
    // has probability ~2^-128 per pair. Memory: 32 bytes per unique frame.
    int dedup;                    // -dedup option, one of MIKU_DEDUP_*
    struct AVHashContext *hash;   // murmur3, allocated in init() for global dedup / delta mode
    MikuTable      dedup_t;       // digest -> file offset of the first copy
    int64_t        dedup_hits;    // statistics, logged at trailer time
    int64_t        dedup_bytes_saved;
    int            any_ref;       // 1 once any frame used a long-term reference

    // Long-term references arriving from a demuxer (stream copy) are named by
    // the SOURCE file's frame numbers, which stop meaning anything the moment
    // the copy drops, trims or reorders frames -- and a stale number can point
    // at some other keyframe and silently produce a corrupt file. Those
    // packets also carry the referenced keyframe's own bytes, so the muxer
    // resolves them by content instead: kf_t maps every keyframe written (in
    // delta mode) to its index in THIS output, and a reference to content not
    // seen yet (a forward reference) waits in pend until write_trailer.
    // References from the encoder are exact output numbers and used as is.
    MikuTable        kf_t;
    MikuPendingRef  *pend;
    int              pend_n, pend_cap;

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
                              uint32_t size, uint32_t duration, int32_t ref)
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
    c->index[c->index_len].ref      = ref;
    c->index_len++;
    return 0;
}

// ---- content tables (global dedup, keyframe map) ----------------------------

static MikuDedupSlot *miku_table_probe(MikuTable *t, const uint8_t *digest,
                                        uint32_t size)
{
    uint32_t mask = t->cap - 1;
    uint32_t i    = (uint32_t)AV_RL64(digest) & mask;

    // Linear probing; the table is kept at most half full, so this always
    // terminates at an empty slot. Returns either the matching slot or the
    // empty slot where this digest would be inserted.
    while (t->slots[i].used) {
        if (t->slots[i].size == size &&
            !memcmp(t->slots[i].digest, digest, 16))
            return &t->slots[i];
        i = (i + 1) & mask;
    }
    return &t->slots[i];
}

static int miku_table_grow(MikuTable *t)
{
    uint32_t old_cap     = t->cap;
    MikuDedupSlot *old   = t->slots;
    uint32_t new_cap     = old_cap ? old_cap * 2 : 4096;
    MikuDedupSlot *fresh = av_calloc(new_cap, sizeof(*fresh));

    if (!fresh)
        return AVERROR(ENOMEM);
    t->slots = fresh;
    t->cap   = new_cap;
    for (uint32_t i = 0; i < old_cap; i++)
        if (old[i].used)
            *miku_table_probe(t, old[i].digest, old[i].size) = old[i];
    av_free(old);
    return 0;
}

// Returns the slot holding this content if it was seen before, else NULL.
static const MikuDedupSlot *miku_table_lookup(MikuTable *t,
                                               const uint8_t *digest, uint32_t size)
{
    MikuDedupSlot *slot;
    if (!t->cap)
        return NULL;
    slot = miku_table_probe(t, digest, size);
    return slot->used ? slot : NULL;
}

// First writer wins: re-inserting content that is already present keeps the
// original value (the earliest copy / the lowest-numbered keyframe).
static int miku_table_insert(MikuTable *t, const uint8_t *digest,
                              uint32_t size, int64_t value)
{
    MikuDedupSlot *slot;
    if ((t->n + 1) * 2 > t->cap) {
        int ret = miku_table_grow(t);
        if (ret < 0)
            return ret;
    }
    slot = miku_table_probe(t, digest, size);
    if (!slot->used) {
        memcpy(slot->digest, digest, 16);
        slot->size  = size;
        slot->value = value;
        slot->used  = 1;
        t->n++;
    }
    return 0;
}

static void miku_table_free(MikuTable *t)
{
    av_freep(&t->slots);
    t->cap = t->n = 0;
}

// 128-bit murmur3 of a byte range, via the context's shared hash.
static void miku_digest(MikuMuxContext *c, const uint8_t *data, size_t size,
                         uint8_t digest[16])
{
    av_hash_init(c->hash);
    av_hash_update(c->hash, data, size);
    av_hash_final_bin(c->hash, digest, 16);
}

// How a packet names its long-term reference, if it has one.
enum {
    MIKU_REF_NONE = 0,
    MIKU_REF_NUMBER,    // encoder: the exact frame number in THIS output
    MIKU_REF_CONTENT,   // demuxer (stream copy): source numbering + the keyframe's bytes
};

// Two sources. The encoder tags a delta packet with AV_PKT_DATA_MIKU_IS_DELTA
// whose 8-byte payload is the reference keyframe's frame number -- exact,
// because the encoder numbers frames in the order the muxer receives them. A
// demuxer tags a packet it hands out for stream copy with
// AV_PKT_DATA_MIKU_REF_FRAME: the SOURCE file's frame number, then the
// keyframe's own bytes; the number is only a hint (see kf_t above), the bytes
// are what the muxer matches on.
static int miku_packet_ref(const AVPacket *pkt, int64_t *number,
                            const uint8_t **kf_data, size_t *kf_size)
{
    size_t sz = 0;
    const uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, &sz);
    if (sd && sz >= 8) {
        *number = (int64_t)AV_RL64(sd);
        return MIKU_REF_NUMBER;
    }
    sd = av_packet_get_side_data(pkt, AV_PKT_DATA_MIKU_REF_FRAME, &sz);
    if (sd && sz > 8 && sz - 8 <= INT32_MAX) {
        *number  = (int64_t)AV_RL64(sd);
        *kf_data = sd + 8;
        *kf_size = sz - 8;
        return MIKU_REF_CONTENT;
    }
    if (sd && sz == 8) { // number only, no bytes to match on: trust it as is
        *number = (int64_t)AV_RL64(sd);
        return MIKU_REF_NUMBER;
    }
    return MIKU_REF_NONE;
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

// Writes dict as a flat JSON object {"key":"value",...}. Shared by the
// format-level metadata, each stream's metadata, and each chapter's
// metadata below -- all exactly this same shape.
static void miku_write_dict_json(AVIOContext *dyn, const AVDictionary *dict)
{
    const AVDictionaryEntry *e = NULL;
    int first = 1;
    avio_w8(dyn, '{');
    while ((e = av_dict_iterate(dict, e))) {
        if (!first)
            avio_w8(dyn, ',');
        first = 0;
        miku_write_json_string(dyn, e->key);
        avio_w8(dyn, ':');
        miku_write_json_string(dyn, e->value);
    }
    avio_w8(dyn, '}');
}

// Serializes s->metadata, per-stream metadata, and chapters into one JSON
// object. Top-level keys are s->metadata verbatim, unchanged from before --
// item 7's original "-metadata key=value" route, so existing readers of the
// old flat shape see no difference. Stream metadata and chapters are new,
// namespaced under __miku_streams/__miku_chapters so they can't collide
// with a real metadata key (a caller would have to go out of their way with
// "-metadata __miku_chapters=...") and so a reader that only understands
// the old flat shape just sees two extra, ignorable top-level keys rather
// than anything that could misparse. Returns a buffer the caller must
// av_free().
static int miku_build_metadata_json(AVFormatContext *s, AVStream *vst, AVStream *ast,
                                     uint8_t **out_buf, int *out_size)
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

    if (!first)
        avio_w8(dyn, ',');
    miku_write_json_string(dyn, "__miku_streams");
    avio_w8(dyn, ':');
    avio_w8(dyn, '{');
    miku_write_json_string(dyn, "video");
    avio_w8(dyn, ':');
    miku_write_dict_json(dyn, vst->metadata);
    avio_w8(dyn, ',');
    miku_write_json_string(dyn, "audio");
    avio_w8(dyn, ':');
    miku_write_dict_json(dyn, ast->metadata);
    avio_w8(dyn, '}');

    if (s->nb_chapters > 0) {
        avio_w8(dyn, ',');
        miku_write_json_string(dyn, "__miku_chapters");
        avio_w8(dyn, ':');
        avio_w8(dyn, '[');
        for (int i = 0; i < s->nb_chapters; i++) {
            AVChapter *ch = s->chapters[i];
            if (i)
                avio_w8(dyn, ',');
            avio_w8(dyn, '{');
            miku_write_json_string(dyn, "tb_num");
            avio_w8(dyn, ':');
            avio_printf(dyn, "%d", ch->time_base.num);
            avio_w8(dyn, ',');
            miku_write_json_string(dyn, "tb_den");
            avio_w8(dyn, ':');
            avio_printf(dyn, "%d", ch->time_base.den);
            avio_w8(dyn, ',');
            miku_write_json_string(dyn, "start");
            avio_w8(dyn, ':');
            avio_printf(dyn, "%lld", (long long)ch->start);
            avio_w8(dyn, ',');
            miku_write_json_string(dyn, "end");
            avio_w8(dyn, ':');
            avio_printf(dyn, "%lld", (long long)ch->end);
            avio_w8(dyn, ',');
            miku_write_json_string(dyn, "metadata");
            avio_w8(dyn, ':');
            miku_write_dict_json(dyn, ch->metadata);
            avio_w8(dyn, '}');
        }
        avio_w8(dyn, ']');
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

    ret = miku_build_metadata_json(s, vst, ast, &json, &json_size);
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

    if (c->dedup == MIKU_DEDUP_GLOBAL || c->delta_mode) {
        // 128-bit murmur3: fast (several GB/s) and non-cryptographic, which
        // is fine -- the input is the encoder's own output, not an adversary's.
        // Delta mode needs it even without global dedup, to match stream-
        // copied long-term references to keyframes by content.
        int ret = av_hash_alloc(&c->hash, "murmur3");
        if (ret < 0)
            return ret;
    }

    c->waiting_thumb = c->thumb_stream_idx >= 0 || c->auto_thumbnail;
    return 0;
}

static int miku_write_header(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    // No attached-pic stream declared AND auto_thumbnail is off: nothing to
    // wait for, write the real header immediately. Otherwise defer --
    // miku_resolve_header runs from write_packet once the explicit
    // thumbnail packet arrives, or (auto_thumbnail's case) once the first
    // video packet does, or once proof there isn't one shows up.
    if (!c->waiting_thumb)
        return miku_flush_header_and_write(s);
    return 0;
}

// Item 3 (dedup) + item 4 (index) + item 2 (VFR durations).
//
// Dedup is two-level. (1) Exact fast path: if this packet is byte-identical
// to the previous one (checked with memcmp, no hashing), reuse its copy --
// the overwhelmingly common case for static or unchanged content. (2) Global
// path (-dedup global, the default): otherwise hash the packet and look it
// up among every unique packet already written, so repeats that are NOT
// adjacent (a looping animation, a scene that returns, a repeated title
// card) are also stored once. Direction does not matter: whichever copy
// appears first in the file is the physical one and every later identical
// packet -- from any later frame -- points back at it.
//
// Dedup works purely on bytes and never changes what a frame IS: each index
// entry carries its own keyframe bit and reference, so two frames can share
// bytes (say, an all-zero residual) while having different roles.
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

    // What this frame IS comes first, because it also decides what has to be
    // hashed below.
    //
    // Delta mode steals size's MSB for the keyframe flag rather than growing
    // the index entry (WebP/delta payloads won't remotely approach 2GB). This
    // only ever happens when delta_mode is set (video codec_id ==
    // AV_CODEC_ID_MIKU_DELTA); with delta_mode off, size_field == size
    // exactly, unchanged from before delta mode existed.
    //
    // The bit comes from *this* packet's own role, never from whatever frame
    // its bytes were deduped against -- offset/size say where the bytes live,
    // this says what frame N itself is.
    //
    // Encoder-origin packets: pkt->flags & AV_PKT_FLAG_KEY is useless here --
    // verified against libavcodec/encode.c that the generic pipeline
    // force-sets it on every packet this encoder produces (AV_CODEC_ID_WEBP's
    // descriptor is AV_CODEC_PROP_INTRA_ONLY, decided before the encoder's own
    // init() mutates codec_id), so it is 1 regardless of what the encoder
    // decided. Those packets mark a delta frame with AV_PKT_DATA_MIKU_IS_DELTA
    // side data instead. Demuxer-origin packets (stream-copy remux) are the
    // opposite: the KEY flag is accurate and there is no IS_DELTA side data.
    // So a frame is a delta if it has the side data OR lacks the key flag; for
    // an encoder packet that is exactly the side data test, for a demuxer
    // packet exactly the flag test.
    int64_t ref_number = -1;
    const uint8_t *ref_kf = NULL;
    size_t ref_kf_size = 0;
    int ref_kind = c->delta_mode ? miku_packet_ref(pkt, &ref_number, &ref_kf, &ref_kf_size)
                                 : MIKU_REF_NONE;
    int is_delta = c->delta_mode &&
        (av_packet_get_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, NULL) != NULL ||
         ref_kind != MIKU_REF_NONE || !(pkt->flags & AV_PKT_FLAG_KEY));

    int is_dup = c->dedup != MIKU_DEDUP_OFF &&
                 c->last_frame_size > 0 && pkt->size > 0 &&
                 c->last_frame_size == pkt->size &&
                 memcmp(c->last_frame_data, pkt->data, pkt->size) == 0;
    uint8_t digest[16];
    int have_digest = 0;
    uint32_t size_field;
    int32_t ref = -1;
    int ret;

    if (is_dup) {
        offset = c->last_frame_offset;
        size   = (uint32_t)c->last_frame_size;
        c->dedup_hits++;
        c->dedup_bytes_saved += pkt->size;
    } else {
        const MikuDedupSlot *hit = NULL;

        if (c->dedup == MIKU_DEDUP_GLOBAL && pkt->size > 0 && c->hash) {
            miku_digest(c, pkt->data, pkt->size, digest);
            have_digest = 1;
            hit = miku_table_lookup(&c->dedup_t, digest, (uint32_t)pkt->size);
        }

        // Whether this packet turns out new or a global repeat, remember its
        // bytes as "the previous frame" so the next packet can try the
        // memcmp fast path against it.
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

        if (hit) {
            offset = hit->value;
            size   = (uint32_t)hit->size;
            c->dedup_hits++;
            c->dedup_bytes_saved += pkt->size;
        } else {
            offset = avio_tell(s->pb);
            avio_write(s->pb, pkt->data, pkt->size);
            size = (uint32_t)pkt->size;
            if (have_digest) {
                ret = miku_table_insert(&c->dedup_t, digest, size, offset);
                if (ret < 0)
                    return ret;
            }
        }
        c->last_frame_offset = offset;
    }

    size_field = size;
    if (c->delta_mode && !is_delta) {
        size_field |= MIKU_KEYFRAME_BIT;
        // Remember which frame of the OUTPUT holds this keyframe's content,
        // so that stream-copied references to it can be matched by content
        // (see kf_t). Costs one hash per keyframe -- free when global dedup
        // already computed it, and keyframes are the rare packets anyway.
        if (pkt->size > 0 && c->hash) {
            if (!have_digest)
                miku_digest(c, pkt->data, pkt->size, digest);
            ret = miku_table_insert(&c->kf_t, digest, (uint32_t)pkt->size, c->index_len);
            if (ret < 0)
                return ret;
        }
    }

    if (is_delta && ref_kind != MIKU_REF_NONE) {
        c->any_ref = 1;
        if (ref_kind == MIKU_REF_CONTENT && c->hash) {
            uint8_t rd[16];
            const MikuDedupSlot *k;

            miku_digest(c, ref_kf, ref_kf_size, rd);
            k = miku_table_lookup(&c->kf_t, rd, (uint32_t)ref_kf_size);
            if (k) {
                ref = (int32_t)k->value;
            } else {
                // Not written yet: a forward reference (or one whose keyframe
                // was cut away, which write_trailer will report). Park it.
                if (c->pend_n == c->pend_cap) {
                    int nc = c->pend_cap ? c->pend_cap * 2 : 16;
                    MikuPendingRef *np = av_realloc_array(c->pend, nc, sizeof(*np));
                    if (!np)
                        return AVERROR(ENOMEM);
                    c->pend = np;
                    c->pend_cap = nc;
                }
                c->pend[c->pend_n].frame = c->index_len;
                c->pend[c->pend_n].size  = (uint32_t)ref_kf_size;
                memcpy(c->pend[c->pend_n].digest, rd, 16);
                c->pend_n++;
                ref = -2; // placeholder, fixed up in write_trailer
            }
        } else {
            if (ref_number < 0 || ref_number > INT32_MAX) {
                av_log(s, AV_LOG_ERROR, "Invalid long-term reference %"PRId64"\n", ref_number);
                return AVERROR(EINVAL);
            }
            ref = (int32_t)ref_number;
        }
    }
    return miku_index_append(c, offset, size_field, duration, ref);
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

// Same role test miku_write_video_packet applies: a packet is a delta frame
// if the encoder tagged it, it names a long-term reference, or (demuxer
// origin) it lacks the key flag.
static int miku_packet_is_delta(const MikuMuxContext *c, const AVPacket *pkt)
{
    int64_t number;
    const uint8_t *kf;
    size_t kf_size;

    return c->delta_mode &&
        (av_packet_get_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, NULL) != NULL ||
         miku_packet_ref(pkt, &number, &kf, &kf_size) != MIKU_REF_NONE ||
         !(pkt->flags & AV_PKT_FLAG_KEY));
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

    // Auto-thumbnail (no explicit attached-pic stream, auto_thumbnail on):
    // the first video packet to arrive doubles as the thumbnail source.
    // Frame 0 is always a real, standalone keyframe (see mikuenc.c), so its
    // bytes are already valid, freestanding WebP data -- a copy, not a
    // re-encode -- and it still gets written as a completely normal frame 0
    // afterward via the fallthrough at the end of this function. The one
    // exception is a stream copy that begins on a delta frame (-copyinkf):
    // that packet is a residual, not a picture, so no thumbnail is better than
    // a noise image.
    if (c->thumb_stream_idx < 0 && c->auto_thumbnail && !c->header_written &&
        pkt->stream_index == c->video_stream_idx) {
        av_freep(&c->thumb_data);
        c->thumb_size = 0;
        if (pkt->size && !miku_packet_is_delta(c, pkt)) {
            c->thumb_data = av_memdup(pkt->data, pkt->size);
            if (!c->thumb_data)
                return AVERROR(ENOMEM);
            c->thumb_size = pkt->size;
        }
        c->waiting_thumb = 0;
        ret = miku_resolve_header(s);
        if (ret < 0)
            return ret;
        return miku_write_video_packet(s, pkt);
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

    // References parked by write_video_packet (stream copy, keyframe not seen
    // yet): every keyframe of the output is known now, so match them by
    // content. A miss means the keyframe was not copied into this output.
    for (i = 0; i < c->pend_n; i++) {
        const MikuDedupSlot *k = miku_table_lookup(&c->kf_t, c->pend[i].digest, c->pend[i].size);
        if (!k) {
            av_log(s, AV_LOG_ERROR,
                   "Frame %d refers to a keyframe that is not part of this output "
                   "(a stream copy that starts or ends between a frame and the "
                   "keyframe it is coded against). Cut at a point where every "
                   "reference stays inside the copy, or re-encode.\n", c->pend[i].frame);
            return AVERROR(EINVAL);
        }
        c->index[c->pend[i].frame].ref = (int32_t)k->value;
    }

    // Long-term references must point at a real keyframe of this same file
    // (any position: earlier or later -- a keyframe is a standalone image).
    // Checked here, with the whole index known, so that a reference to a
    // later frame is also verifiable; anything else would be a file that
    // decodes to garbage, so refuse to finish it.
    for (i = 0; i < c->index_len; i++) {
        int32_t ref = c->index[i].ref;
        if (ref < 0)
            continue;
        if (ref >= c->index_len || !(c->index[ref].size & MIKU_KEYFRAME_BIT) ||
            (c->index[i].size & MIKU_KEYFRAME_BIT)) {
            av_log(s, AV_LOG_ERROR,
                   "Frame %d has a long-term reference to frame %d, which is "
                   "not a keyframe in this file (were packets dropped or "
                   "trimmed between the encoder and the muxer?)\n", i, ref);
            return AVERROR(EINVAL);
        }
    }

    if (c->audio_dyn)
        audio_size = avio_close_dyn_buf(c->audio_dyn, &audio_buf);
    c->audio_dyn = NULL;

    audio_offset = avio_tell(s->pb);
    if (audio_buf && audio_size > 0)
        avio_write(s->pb, audio_buf, audio_size);
    av_free(audio_buf);

    // Item 4: the real index, written now that every frame's offset/size and
    // the audio blob's location are finally known. Entry shape depends on
    // whether any frame used a long-term reference (see the layout comment
    // at the top of this file): plain 16-byte entries keep the file readable
    // by older demuxers; 24-byte entries carry the reference.
    index_offset = avio_tell(s->pb);
    for (i = 0; i < c->index_len; i++) {
        avio_wl64(s->pb, c->index[i].offset);
        avio_wl32(s->pb, c->index[i].size);
        avio_wl32(s->pb, c->index[i].duration);
        if (c->any_ref) {
            avio_wl32(s->pb, (uint32_t)c->index[i].ref);
            avio_wl32(s->pb, 0); // reserved
        }
    }
    audio_samples_u32 = c->audio_total_samples > 0xFFFFFFFFLL
                             ? 0xFFFFFFFFU : (uint32_t)c->audio_total_samples;
    avio_wl64(s->pb, audio_offset);
    avio_wl32(s->pb, (uint32_t)audio_size);
    avio_wl32(s->pb, audio_samples_u32); // mikudec.c doesn't read this field today; kept for forward compatibility
    if (c->any_ref)
        avio_wl64(s->pb, 0); // pad the audio entry out to the v3 stride

    end_pos = avio_tell(s->pb);

    // Seek back and patch the header fields that couldn't be known up front
    // (item 4) -- and, only if a long-term reference was actually used, the
    // format version and codec flag, so old readers refuse the file instead
    // of mis-decoding it.
    avio_seek(s->pb, MIKU_OFF_FRAME_COUNT, SEEK_SET);
    avio_wl32(s->pb, (uint32_t)c->index_len);
    avio_seek(s->pb, MIKU_OFF_INDEX_OFF, SEEK_SET);
    avio_wl64(s->pb, index_offset);
    if (c->any_ref) {
        avio_seek(s->pb, MIKU_OFF_VERSION, SEEK_SET);
        avio_wl16(s->pb, MIKU_FORMAT_VERSION_REFS);
        avio_seek(s->pb, MIKU_OFF_CODEC_FLAG, SEEK_SET);
        avio_w8(s->pb, MIKU_CODEC_WEBP_DELTA_REFS);
    }
    avio_seek(s->pb, end_pos, SEEK_SET);

    av_log(s, AV_LOG_VERBOSE,
           "miku: %d frames, %"PRId64" deduplicated (%"PRId64" bytes not written), "
           "format v%d\n", c->index_len, c->dedup_hits, c->dedup_bytes_saved,
           c->any_ref ? MIKU_FORMAT_VERSION_REFS
                      : c->delta_mode ? MIKU_FORMAT_VERSION_DELTA : MIKU_FORMAT_VERSION);

    return 0;
}

static void miku_deinit(AVFormatContext *s)
{
    MikuMuxContext *c = s->priv_data;
    uint8_t *tmp;

    av_freep(&c->index);
    av_freep(&c->last_frame_data);
    av_freep(&c->thumb_data);
    miku_table_free(&c->dedup_t);
    miku_table_free(&c->kf_t);
    av_freep(&c->pend);
    c->pend_n = c->pend_cap = 0;
    av_hash_freep(&c->hash);
    if (c->audio_dyn) {
        avio_close_dyn_buf(c->audio_dyn, &tmp);
        av_free(tmp);
        c->audio_dyn = NULL;
    }
    ff_packet_list_free(&c->queue);
}

#define MOFFSET(x) offsetof(MikuMuxContext, x)
#define ME AV_OPT_FLAG_ENCODING_PARAM
static const AVOption miku_mux_options[] = {
    { "auto_thumbnail", "If no attached-picture thumbnail stream is provided (e.g. via "
      "-map 1 -disposition:v:1 attached_pic), use the first video keyframe's own bytes "
      "as one automatically instead of shipping with no thumbnail. Free: frame 0 is "
      "already a real, standalone WebP keyframe by construction (see mikuenc.c), so this "
      "is a byte copy, not a re-encode. Set to 0 to skip thumbnails entirely when none "
      "was explicitly provided.",
      MOFFSET(auto_thumbnail), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, ME },
    { "dedup", "Store byte-identical video packets only once. 'global' (default) reuses "
      "the first copy for ANY later identical packet anywhere in the file -- loops, "
      "returning scenes, repeated slates; 'last' only compares against the immediately "
      "preceding packet (the old behavior, no hash table); 'off' stores everything. "
      "Dedup never changes decoded output, only file size.",
      MOFFSET(dedup), AV_OPT_TYPE_INT, { .i64 = MIKU_DEDUP_GLOBAL }, MIKU_DEDUP_OFF, MIKU_DEDUP_GLOBAL, ME, .unit = "dedup" },
        { "off",    "no dedup",                       0, AV_OPT_TYPE_CONST, { .i64 = MIKU_DEDUP_OFF },    0, 0, ME, .unit = "dedup" },
        { "last",   "against the previous packet",    0, AV_OPT_TYPE_CONST, { .i64 = MIKU_DEDUP_LAST },   0, 0, ME, .unit = "dedup" },
        { "global", "against every earlier packet",   0, AV_OPT_TYPE_CONST, { .i64 = MIKU_DEDUP_GLOBAL }, 0, 0, ME, .unit = "dedup" },
    { NULL },
};

static const AVClass miku_muxer_class = {
    .class_name = "miku",
    .item_name  = av_default_item_name,
    .option     = miku_mux_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFOutputFormat ff_miku_muxer = {
    .p.name         = "miku",
    .p.long_name    = NULL_IF_CONFIG_SMALL("MIKU Lossless Video"),
    .p.extensions   = "miku",
    // v1 is WebP-only (init() rejects anything else); WebP is also the
    // default a caller gets if -c:v isn't passed explicitly.
    .p.audio_codec  = AV_CODEC_ID_FLAC,
    .p.video_codec  = AV_CODEC_ID_WEBP,
    .p.flags        = AVFMT_GLOBALHEADER | AVFMT_VARIABLE_FPS,
    .p.priv_class   = &miku_muxer_class,
    .priv_data_size = sizeof(MikuMuxContext),
    .init           = miku_init,
    .write_header   = miku_write_header,
    .write_packet   = miku_write_packet,
    .write_trailer  = miku_write_trailer,
    .deinit         = miku_deinit,
};
