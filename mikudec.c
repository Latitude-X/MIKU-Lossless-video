// mikudec.c — .miku demuxer
//
// Container: magic "MIKU", 64-byte header, META block, the video frames, one
// FLAC audio blob and a per-frame index table: 16 bytes per entry {offset u64,
// size u32, duration u32}, followed by one trailing entry for the audio blob.
// A video frame's payload is a lossless WebP image; the audio blob is a
// complete embedded FLAC stream. mikumux.c documents the full layout and the
// header field offsets.
//
// The codec flag selects the video coding and the index shape: 0x01 (WebP),
// 0x02 (WebP delta) or 0x03 (WebP delta with long-term references, below);
// no other value is accepted. In delta files (0x02, 0x03) the MSB of each
// video entry's size field is a keyframe flag, see miku_get_index_entry. In
// plain WebP files every frame is a keyframe.
//
// Format v3 (codec flag 0x03) adds LONG-TERM REFERENCES: a delta frame may
// reconstruct against any keyframe in the file instead of the previous frame,
// and the index entries widen to 24 bytes {offset u64, size u32, duration u32,
// ref i32, reserved u32} to say which. Because a keyframe is a standalone
// image, the referenced keyframe can sit anywhere, before or after the frame
// that uses it, so for such a frame this demuxer hands the decoder a
// self-contained packet: the frame's residual as the payload, plus the
// referenced keyframe's bytes as AV_PKT_DATA_MIKU_REF_FRAME side data. The
// decoder needs nothing else, which also makes seeking safe: a frame with a
// reference is a chain root, like a keyframe.

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
#define MIKU_INDEX_ENTRY_SIZE  16 // v1/v2 entry stride; v3 uses MIKU_INDEX_ENTRY_SIZE_V3
#define MIKU_INDEX_ENTRY_SIZE_V3 24
#define MIKU_FORMAT_VERSION_MAX  3
#define MIKU_OFF_VERSION       4
#define MIKU_CODEC_FLAG_OFFSET 45
#define MIKU_OFF_METADATA_OFF  22
#define MIKU_OFF_VIDEO_OFF     46 // needed to size the implicitly-lengthed thumbnail; see miku_read_metadata_block
#define MIKU_OFF_SAR_NUM       54
#define MIKU_OFF_SAR_DEN       56
#define MIKU_META_FLAG_THUMBNAIL 0x02
#define MIKU_META_FLAG_COLOR     0x04
#define MIKU_CODEC_WEBP        0x01
#define MIKU_CODEC_WEBP_DELTA  0x02  // see mikumux.c -- keyframe bit stolen from each
                                      // video index entry's size MSB when this is set
#define MIKU_CODEC_WEBP_DELTA_REFS 0x03 // delta mode + 24-byte entries carrying a
                                         // long-term reference per frame (format v3)

// Above this many frames the index is not preloaded into RAM. One million
// frames cost 16 to 21 MB (per frame: offset 8, size 4, duration 4 bytes, plus
// one keyframe byte in delta files and a 4-byte reference in v3), which is
// comfortably safe, but a file with tens of millions of frames could add up,
// so past this point index entries are read from disk on demand.
#define MIKU_INDEX_PRELOAD_MAX 1000000

// Size of one audio read (the last read of the blob may be shorter); also the
// interleaving grain, see miku_read_packet().
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
    int32_t *frame_ref;          // v3 only (NULL otherwise): index of the long-term
                                  // reference keyframe per frame, -1 for none
    int      entry_size;         // index stride: 16 (v1/v2) or 24 (v3)
    int      num_frames;
    int      current_frame;
    int64_t  next_pts;           // pts of the next video packet: the running sum of the
                                  // real per-frame durations. Frame i's pts is the sum
                                  // of the durations of frames [0, i), not i itself:
                                  // durations vary (the format is VFR), so one tick per
                                  // frame would mistime every frame after the first
                                  // whose duration is not 1. See miku_read_video_packet
                                  // and miku_frame_for_pts.

    int      delta_mode;         // 1 iff codec_flag is 0x02 or 0x03 (delta video); gates
                                  // keyframe-bit extraction from the index. When 0, every
                                  // frame is implicitly a keyframe.
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
    // Where the file position is believed to be, so that
    // miku_read_video_packet/miku_read_audio_packet can skip a redundant
    // avio_seek() when the previous read left it in the right place. -1 means
    // "unknown, must seek before the next read."
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

// Parses the embedded FLAC stream's own "fLaC" magic and metadata block chain
// (starting at audio_offset) to find the STREAMINFO block and the byte offset
// where the FLAC *frame* data begins. STREAMINFO is copied verbatim into
// codecpar->extradata, as libavformat/flacdec.c does: FLAC_STREAMINFO_SIZE
// (34) bytes.
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

// ---- Minimal JSON reader for the META block's JSON metadata -------------
//
// Matches exactly what miku_build_metadata_json (mikumux.c) produces:
// miku_write_json_string's escaping, flat objects of string:string pairs,
// one level of nesting under __miku_streams, and an array of objects under
// __miku_chapters. Deliberately not a general-purpose JSON parser -- same
// "match exactly what we write, nothing more" spirit as the writer's own
// hand-rolled string escaper. Any structure this doesn't recognize aborts
// the metadata parse (see miku_parse_metadata_json's own comment for why
// that's fine) rather than trying to skip unknown values generically.

static void miku_json_skip_ws(const char **p)
{
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')
        (*p)++;
}

// Parses a JSON string starting at **p (must point at the opening quote)
// into a newly av_malloc'd, NUL-terminated buffer, and advances *p past the
// closing quote. Returns NULL, leaving *p unchanged, on anything that isn't
// a validly-escaped JSON string this format would ever actually write.
static char *miku_json_parse_string(const char **p)
{
    const char *start, *scan;
    int len = 0;
    char *out, *w;

    if (**p != '"')
        return NULL;
    scan = *p + 1;

    // Pass 1: validate and compute the decoded length.
    while (*scan && *scan != '"') {
        if ((unsigned char)*scan == '\\') {
            scan++;
            switch (*scan) {
            case '"': case '\\': case '/': case 'n': case 'r': case 't':
                len++;
                scan++;
                break;
            case 'u': {
                int i;
                for (i = 1; i <= 4; i++) {
                    char h = scan[i];
                    if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')))
                        return NULL;
                }
                len += 3; // worst-case UTF-8 encoding of one BMP code point
                scan += 5;
                break;
            }
            default:
                return NULL; // not an escape this format ever writes
            }
        } else {
            len++;
            scan++;
        }
    }
    if (*scan != '"')
        return NULL; // ran off the end without a closing quote

    out = av_malloc(len + 1);
    if (!out)
        return NULL;
    w = out;
    start = *p + 1;
    while (*start != '"') {
        if ((unsigned char)*start == '\\') {
            start++;
            switch (*start) {
            case '"':  *w++ = '"';  start++; break;
            case '\\': *w++ = '\\'; start++; break;
            case '/':  *w++ = '/';  start++; break;
            case 'n':  *w++ = '\n'; start++; break;
            case 'r':  *w++ = '\r'; start++; break;
            case 't':  *w++ = '\t'; start++; break;
            case 'u': {
                unsigned cp = 0;
                int i;
                for (i = 1; i <= 4; i++) {
                    char h = start[i];
                    unsigned d = (h >= '0' && h <= '9') ? (unsigned)(h - '0')
                                : (h >= 'a' && h <= 'f') ? (unsigned)(h - 'a' + 10)
                                                          : (unsigned)(h - 'A' + 10);
                    cp = (cp << 4) | d;
                }
                start += 5;
                if (cp < 0x80) {
                    *w++ = (char)cp;
                } else if (cp < 0x800) {
                    *w++ = (char)(0xC0 | (cp >> 6));
                    *w++ = (char)(0x80 | (cp & 0x3F));
                } else {
                    *w++ = (char)(0xE0 | (cp >> 12));
                    *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
                    *w++ = (char)(0x80 | (cp & 0x3F));
                }
                break;
            }
            }
        } else {
            *w++ = *start;
            start++;
        }
    }
    *w = '\0';
    *p = start + 1; // past the closing quote
    return out;
}

// Parses a JSON integer: optional leading '-', digits only -- matches
// exactly what avio_printf("%d"/"%lld", ...) writes for the chapter
// fields, nothing fancier. Returns 0, leaving *p unchanged, on no match.
static int miku_json_parse_int(const char **p, int64_t *out)
{
    const char *s = *p;
    int neg = 0;
    int64_t v = 0;
    int any = 0;

    if (*s == '-') { neg = 1; s++; }
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (*s - '0');
        s++;
        any = 1;
    }
    if (!any)
        return 0;
    *out = neg ? -v : v;
    *p = s;
    return 1;
}

// Parses a flat {"key":"value",...} object starting at **p (opening brace)
// and applies each entry to *dict via av_dict_set. Advances *p past the
// closing brace. Returns <0, leaving *dict partially populated, on
// malformed input -- callers treat that as "abandon the whole metadata
// parse", so a partial dict never actually reaches the caller.
static int miku_json_parse_dict(const char **p, AVDictionary **dict)
{
    miku_json_skip_ws(p);
    if (**p != '{')
        return AVERROR_INVALIDDATA;
    (*p)++;
    miku_json_skip_ws(p);
    if (**p == '}') { (*p)++; return 0; }

    for (;;) {
        char *key, *val;
        miku_json_skip_ws(p);
        key = miku_json_parse_string(p);
        if (!key)
            return AVERROR_INVALIDDATA;
        miku_json_skip_ws(p);
        if (**p != ':') { av_free(key); return AVERROR_INVALIDDATA; }
        (*p)++;
        miku_json_skip_ws(p);
        val = miku_json_parse_string(p);
        if (!val) { av_free(key); return AVERROR_INVALIDDATA; }
        // DONT_STRDUP_*: dict takes ownership of key/val either way,
        // freeing them itself even if this particular set fails.
        av_dict_set(dict, key, val, AV_DICT_DONT_STRDUP_KEY | AV_DICT_DONT_STRDUP_VAL);
        miku_json_skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == '}') { (*p)++; break; }
        return AVERROR_INVALIDDATA;
    }
    return 0;
}

// Parses the top-level object miku_build_metadata_json writes: flat
// format-metadata keys verbatim, plus __miku_streams (video/audio metadata)
// and __miku_chapters when present. Malformed or unrecognized JSON just
// abandons the rest of the parse instead of failing the read: metadata is
// never load-bearing for whether the file plays, so a corrupt or foreign META
// block should cost the tags and chapters, not the video.
static void miku_parse_metadata_json(AVFormatContext *s, AVStream *vst, AVStream *ast,
                                      const uint8_t *json, int json_len)
{
    char *buf = av_malloc(json_len + 1);
    const char *p;
    if (!buf)
        return;
    memcpy(buf, json, json_len);
    buf[json_len] = '\0';
    p = buf;

    miku_json_skip_ws(&p);
    if (*p != '{')
        goto done;
    p++;
    miku_json_skip_ws(&p);
    if (*p == '}')
        goto done;

    for (;;) {
        char *key;
        miku_json_skip_ws(&p);
        key = miku_json_parse_string(&p);
        if (!key)
            goto done;
        miku_json_skip_ws(&p);
        if (*p != ':') { av_free(key); goto done; }
        p++;
        miku_json_skip_ws(&p);

        if (!strcmp(key, "__miku_streams")) {
            av_free(key);
            miku_json_skip_ws(&p);
            if (*p != '{') goto done;
            p++;
            miku_json_skip_ws(&p);
            if (*p != '}') {
                for (;;) {
                    char *skey;
                    AVDictionary **target;
                    miku_json_skip_ws(&p);
                    skey = miku_json_parse_string(&p);
                    if (!skey) goto done;
                    miku_json_skip_ws(&p);
                    if (*p != ':') { av_free(skey); goto done; }
                    p++;
                    target = !strcmp(skey, "video") ? &vst->metadata
                            : !strcmp(skey, "audio") ? &ast->metadata
                                                      : NULL;
                    av_free(skey);
                    if (!target || miku_json_parse_dict(&p, target) < 0)
                        goto done;
                    miku_json_skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == '}') { p++; break; }
                    goto done;
                }
            } else {
                p++;
            }
        } else if (!strcmp(key, "__miku_chapters")) {
            av_free(key);
            miku_json_skip_ws(&p);
            if (*p != '[') goto done;
            p++;
            miku_json_skip_ws(&p);
            if (*p != ']') {
                for (;;) {
                    int64_t tb_num = 1, tb_den = 1, start = 0, end = 0;
                    AVDictionary *chmeta = NULL;
                    AVChapter *ch;
                    const AVDictionaryEntry *title_e;

                    miku_json_skip_ws(&p);
                    if (*p != '{') goto done;
                    p++;
                    miku_json_skip_ws(&p);
                    if (*p != '}') {
                        for (;;) {
                            char *fkey;
                            int64_t ival;
                            miku_json_skip_ws(&p);
                            fkey = miku_json_parse_string(&p);
                            if (!fkey) { av_dict_free(&chmeta); goto done; }
                            miku_json_skip_ws(&p);
                            if (*p != ':') { av_free(fkey); av_dict_free(&chmeta); goto done; }
                            p++;
                            miku_json_skip_ws(&p);
                            if (!strcmp(fkey, "metadata")) {
                                if (miku_json_parse_dict(&p, &chmeta) < 0) {
                                    av_free(fkey); av_dict_free(&chmeta); goto done;
                                }
                            } else if (!strcmp(fkey, "tb_num") || !strcmp(fkey, "tb_den") ||
                                       !strcmp(fkey, "start")  || !strcmp(fkey, "end")) {
                                if (!miku_json_parse_int(&p, &ival)) {
                                    av_free(fkey); av_dict_free(&chmeta); goto done;
                                }
                                if      (!strcmp(fkey, "tb_num")) tb_num = ival;
                                else if (!strcmp(fkey, "tb_den")) tb_den = ival;
                                else if (!strcmp(fkey, "start"))  start  = ival;
                                else                               end    = ival;
                            } else {
                                av_free(fkey); av_dict_free(&chmeta); goto done;
                            }
                            av_free(fkey);
                            miku_json_skip_ws(&p);
                            if (*p == ',') { p++; continue; }
                            if (*p == '}') { p++; break; }
                            av_dict_free(&chmeta);
                            goto done;
                        }
                    } else {
                        p++;
                    }

                    if (tb_den <= 0) tb_den = 1;
                    title_e = av_dict_get(chmeta, "title", NULL, 0);
                    ch = avpriv_new_chapter(s, s->nb_chapters,
                                             (AVRational){ (int)tb_num, (int)tb_den },
                                             start, end, title_e ? title_e->value : NULL);
                    if (ch) {
                        const AVDictionaryEntry *me = NULL;
                        while ((me = av_dict_iterate(chmeta, me)))
                            if (strcmp(me->key, "title")) // new_chapter already set title
                                av_dict_set(&ch->metadata, me->key, me->value, 0);
                    }
                    av_dict_free(&chmeta);

                    miku_json_skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    if (*p == ']') { p++; break; }
                    goto done;
                }
            } else {
                p++;
            }
        } else {
            char *val = miku_json_parse_string(&p);
            if (!val) { av_free(key); goto done; }
            av_dict_set(&s->metadata, key, val, AV_DICT_DONT_STRDUP_VAL);
            av_free(key);
        }

        miku_json_skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; break; }
        goto done;
    }

done:
    av_free(buf);
}


// Restores the colour/HDR metadata onto vst->codecpar. WebP's bitstream has no
// field for it, so the muxer stores it at the container level, and this
// mirrors how libavformat/movenc.c and mov.c handle mdcv/clli. Only the color
// sub-block is parsed here; the JSON metadata and any thumbnail are handled by
// miku_read_metadata_block below, once both streams exist.
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

// Companion to miku_parse_meta_block above: re-seeks to the same META
// block and reads its JSON portion for real, instead of skipping over it.
// Split into a separate pass, called later in miku_read_header once both
// streams exist (miku_parse_meta_block runs while only vst does, since the
// color sub-block it handles only ever needed vst) -- the tiny duplicate
// seek+read of the block header is negligible, this runs once per open.
// Never fails the open: metadata is never load-bearing for whether the
// file plays, so any problem here just means no metadata, not an error.
static int miku_read_metadata_block(AVFormatContext *s, AVStream *vst, AVStream *ast,
                                     int64_t metadata_offset, int64_t video_offset)
{
    AVIOContext *pb = s->pb;
    uint8_t magic[4];
    uint8_t flags;
    uint32_t json_len;
    uint8_t *json;
    int ret;

    avio_seek(pb, metadata_offset, SEEK_SET);
    ret = avio_read(pb, magic, 4);
    if (ret != 4 || memcmp(magic, "META", 4) != 0)
        return 0; // no META block (older file, or offset unset)
    flags = avio_r8(pb);
    json_len = avio_rl32(pb);
    if (json_len == 0 || json_len > 64 * 1024 * 1024) { // 0 = none; cap against a corrupt/hostile length
        avio_skip(pb, json_len);
    } else {
        json = av_malloc(json_len);
        if (json) {
            ret = avio_read(pb, json, json_len);
            if (ret == (int)json_len)
                miku_parse_metadata_json(s, vst, ast, json, json_len);
            // else: truncated META block -- treat as no metadata, not a hard failure
            av_free(json);
        }
    }

    // Thumbnail exposure. Structurally skip the color sub-block if present
    // -- miku_parse_meta_block elsewhere in this file already extracts its
    // actual values on its own separate pass; this one only needs to land
    // at the right byte offset. Mirrors mikumux.c's own layout comment:
    // "META"+json, [color block], [thumbnail], each self-describing except
    // the thumbnail, whose length is implicitly "everything up to
    // video_offset" (see that file for why it's written that way).
    if (flags & MIKU_META_FLAG_COLOR) {
        avio_skip(pb, 4); // primaries, trc, space, range
        if (avio_r8(pb)) avio_skip(pb, avio_rl32(pb)); // mdcv, if present
        if (avio_r8(pb)) avio_skip(pb, avio_rl32(pb)); // clli, if present
    }
    if (flags & MIKU_META_FLAG_THUMBNAIL) {
        int64_t thumb_offset = avio_tell(pb);
        int64_t thumb_size   = video_offset - thumb_offset;
        // Real, standalone WebP bytes either way -- auto-generated
        // thumbnails are frame 0's own bytes verbatim (see mikumux.c's
        // write_packet), explicitly-provided ones are whatever the caller
        // handed the thumbnail stream, itself required to already be a
        // complete image. Exposed the standard FFmpeg way: a stream
        // carrying AV_DISPOSITION_ATTACHED_PIC, decodable like any other
        // single WebP image.
        if (thumb_size > 0 && thumb_size <= 256 * 1024 * 1024) {
            int ares = ff_add_attached_pic(s, NULL, pb, NULL, (int)thumb_size);
            if (ares >= 0)
                s->streams[s->nb_streams - 1]->codecpar->codec_id = AV_CODEC_ID_WEBP;
            // else: non-fatal, same "metadata/extras never block the video
            // itself" policy as everywhere else in this function
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
    uint8_t entry[MIKU_INDEX_ENTRY_SIZE_V3];
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
    if (AV_RL16(header + MIKU_OFF_VERSION) > MIKU_FORMAT_VERSION_MAX) {
        av_log(s, AV_LOG_ERROR, "Unsupported .miku format version %d (this build reads up to %d).\n",
               AV_RL16(header + MIKU_OFF_VERSION), MIKU_FORMAT_VERSION_MAX);
        return AVERROR_PATCHWELCOME;
    }

    // ---- Video stream ----
    vst = avformat_new_stream(s, NULL);
    if (!vst) return AVERROR(ENOMEM);
    vst->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

    // Only WebP (0x01), WebP delta (0x02) and WebP delta with long-term
    // references (0x03) are accepted; there is no codec flag 0x00 (PNG).
    if (codec_flag != MIKU_CODEC_WEBP && codec_flag != MIKU_CODEC_WEBP_DELTA &&
        codec_flag != MIKU_CODEC_WEBP_DELTA_REFS) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported .miku codec flag 0x%02x (PNG support was removed; "
               "only WebP (0x01), WebP-delta (0x02) or WebP-delta with long-term "
               "references (0x03) are accepted).\n", codec_flag);
        return AVERROR_INVALIDDATA;
    }
    m->delta_mode = (codec_flag == MIKU_CODEC_WEBP_DELTA ||
                     codec_flag == MIKU_CODEC_WEBP_DELTA_REFS);
    m->entry_size = (codec_flag == MIKU_CODEC_WEBP_DELTA_REFS) ? MIKU_INDEX_ENTRY_SIZE_V3
                                                               : MIKU_INDEX_ENTRY_SIZE;
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

    // ---- Index table: preloaded up to MIKU_INDEX_PRELOAD_MAX frames, else read on demand ----
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
        if (m->entry_size == MIKU_INDEX_ENTRY_SIZE_V3) {
            m->frame_ref = av_malloc_array(m->num_frames, sizeof(int32_t));
            if (!m->frame_ref)
                return AVERROR(ENOMEM);
        }

        avio_seek(pb, idx_off, SEEK_SET);
        for (int i = 0; i < m->num_frames; i++) {
            uint32_t raw_size;
            ret = avio_read(pb, entry, m->entry_size);
            if (ret != m->entry_size) return AVERROR_INVALIDDATA;
            m->frame_offsets[i]   = AV_RL64(entry);
            raw_size              = AV_RL32(entry + 8);
            m->frame_durations[i] = AV_RL32(entry + 12);
            if (m->delta_mode) {
                m->frame_keyframe[i] = (raw_size & 0x80000000u) ? 1 : 0;
                m->frame_sizes[i]    = (int32_t)(raw_size & 0x7FFFFFFFu);
            } else {
                m->frame_sizes[i]    = (int32_t)raw_size;
            }
            if (m->frame_ref)
                m->frame_ref[i] = (int32_t)AV_RL32(entry + 16);
        }

        // Every long-term reference must name a keyframe of this file, and
        // only a delta frame may carry one. Verified once, up front, so a
        // corrupt index fails the open instead of misdecoding mid-playback
        // (the streaming path below re-checks each reference as it is used).
        if (m->frame_ref) {
            for (int i = 0; i < m->num_frames; i++) {
                int32_t r = m->frame_ref[i];
                if (r < 0)
                    continue;
                if (r >= m->num_frames || !m->frame_keyframe[r] || m->frame_keyframe[i]) {
                    av_log(s, AV_LOG_ERROR,
                           "Corrupt index: frame %d has an invalid long-term reference (%d)\n", i, r);
                    return AVERROR_INVALIDDATA;
                }
            }
        }
    } else {
        // Streaming-index fallback: read each entry from disk on demand in
        // miku_get_index_entry() instead of allocating O(num_frames) memory.
        m->index_in_memory    = 0;
        m->index_table_offset = idx_off;
    }

    // ---- Trailing index entry: audio blob location/size ----
    audio_idx_pos = idx_off + (int64_t)m->num_frames * m->entry_size;
    avio_seek(pb, audio_idx_pos, SEEK_SET);
    ret = avio_read(pb, entry, m->entry_size);
    if (ret != m->entry_size) return AVERROR_INVALIDDATA;
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

    // FLAC is compressed and frame-based, so the blob cannot be cut at
    // arbitrary byte boundaries into decodable packets. Instead:
    //   1. Parse past "fLaC" and the metadata blocks to find where the frame
    //      data starts, and copy STREAMINFO into extradata
    //      (miku_parse_flac_header; mirrors libavformat/flacdec.c).
    //   2. Mark the stream AVSTREAM_PARSE_FULL_RAW so libavformat's FLAC
    //      parser re-splits whatever byte chunks it is handed into correctly
    //      bounded and timestamped frames (flacdec.c does the same).
    ret = miku_parse_flac_header(m, pb, ast, audio_offset, audio_offset + (int64_t)audio_size);
    if (ret < 0) return ret;
    m->audio_frame_end = audio_offset + audio_size;
    ffstream(ast)->need_parsing = AVSTREAM_PARSE_FULL_RAW;

    // Format/stream metadata + chapters: needs both streams to exist (see
    // miku_read_metadata_block's own comment), so it runs here rather than
    // alongside the color sub-block above.
    miku_read_metadata_block(s, vst, ast, AV_RL64(header + MIKU_OFF_METADATA_OFF),
                              AV_RL64(header + MIKU_OFF_VIDEO_OFF));

    // ---- Interleaving plan ----
    // Serving every video packet before any audio packet would starve a
    // player's audio queue on a large file: it fills its bounded video queue
    // and stalls long before demuxing reaches the first audio byte. Instead
    // MIKU_AUDIO_CHUNK_SIZE-byte audio reads are spread evenly across the
    // video frame sequence; see miku_read_packet().
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

// Fetches index entry i whether the table is in memory or read from disk on
// demand. is_keyframe is always 1 when delta_mode is off (every frame stands
// alone); when it is on, it is the MSB taken from size, which is masked out
// here so that no caller, and in particular the "size < 0 is corrupt data"
// check, ever sees the raw flag bit.
static int miku_get_index_entry(AVFormatContext *s, int i,
                                 int64_t *offset, int32_t *size, int32_t *duration,
                                 int *is_keyframe, int32_t *ref)
{
    MikuContext *m = s->priv_data;

    if (m->index_in_memory) {
        *offset      = m->frame_offsets[i];
        *size        = m->frame_sizes[i]; // already masked at load time
        *duration    = m->frame_durations[i];
        *is_keyframe = m->delta_mode ? m->frame_keyframe[i] : 1;
        *ref         = m->frame_ref ? m->frame_ref[i] : -1;
        return 0;
    }

    {
        AVIOContext *pb = s->pb;
        uint8_t entry[MIKU_INDEX_ENTRY_SIZE_V3];
        int64_t pos = m->index_table_offset + (int64_t)i * m->entry_size;
        uint32_t raw_size;
        int ret;

        avio_seek(pb, pos, SEEK_SET);
        ret = avio_read(pb, entry, m->entry_size);
        if (ret != m->entry_size) return AVERROR_INVALIDDATA;
        *offset  = AV_RL64(entry);
        raw_size = AV_RL32(entry + 8);
        *duration = AV_RL32(entry + 12);
        *ref      = m->entry_size == MIKU_INDEX_ENTRY_SIZE_V3 ? (int32_t)AV_RL32(entry + 16) : -1;
        if (m->delta_mode) {
            *is_keyframe = (raw_size & 0x80000000u) ? 1 : 0;
            *size        = (int32_t)(raw_size & 0x7FFFFFFFu);
        } else {
            *is_keyframe = 1;
            *size        = (int32_t)raw_size;
        }
        m->file_pos = pos + m->entry_size; // fd moved; keep the cache honest
        return 0;
    }
}

static int miku_read_video_packet(AVFormatContext *s, AVPacket *pkt)
{
    MikuContext *m = s->priv_data;
    AVIOContext *pb = s->pb;
    int64_t offset;
    int32_t size, duration;
    int32_t ref;
    int is_keyframe;
    int ret;

    ret = miku_get_index_entry(s, m->current_frame, &offset, &size, &duration, &is_keyframe, &ref);
    if (ret < 0) return ret;
    if (size < 0 || offset < 0)
        return AVERROR_INVALIDDATA; // corrupt index entry; don't hand a bogus size to av_new_packet

    // Skip the seek if the file position is already here.
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
    pkt->pts          = m->next_pts; // the sum of the real durations so far, not
                                      // the frame index; see next_pts
    m->next_pts      += pkt->duration;
    // Non-delta files: every frame is an independent image, hence a keyframe.
    // Delta files: the flag follows the keyframe bit the muxer wrote for this
    // frame, which the decoder relies on to know where a chain starts.
    if (is_keyframe)
        pkt->flags |= AV_PKT_FLAG_KEY;

    m->file_pos = offset + size;

    // Long-term reference: make the packet self-contained by attaching the
    // referenced keyframe's own bytes. The keyframe may be anywhere in the
    // file, before or after this frame, so the decoder cannot count on having
    // seen it, and after a seek it has not. The decoder keeps a small cache of
    // decoded references keyed by the keyframe's compressed bytes, so a run of
    // frames sharing one reference decodes it once; re-sending the compressed
    // bytes costs one extra read, normally served from the page cache.
    if (ref >= 0) {
        int64_t roff;
        int32_t rsize, rdur, rref;
        int rkey;
        uint8_t *sd;

        // In-memory files were validated at open; streaming mode (index read
        // on demand) has not been, so bound the reference here before it is
        // used as an index into the table.
        if (ref >= m->num_frames) {
            av_log(s, AV_LOG_ERROR,
                   "Corrupt index: frame %d has an invalid long-term reference (%d)\n",
                   m->current_frame, ref);
            av_packet_unref(pkt);
            return AVERROR_INVALIDDATA;
        }
        ret = miku_get_index_entry(s, ref, &roff, &rsize, &rdur, &rkey, &rref);
        if (ret < 0) {
            av_packet_unref(pkt);
            return ret;
        }
        if (!rkey || is_keyframe || roff < 0 || rsize <= 0) {
            av_log(s, AV_LOG_ERROR,
                   "Corrupt index: frame %d has an invalid long-term reference (%d)\n",
                   m->current_frame, ref);
            av_packet_unref(pkt);
            return AVERROR_INVALIDDATA;
        }
        sd = av_packet_new_side_data(pkt, AV_PKT_DATA_MIKU_REF_FRAME, 8 + (size_t)rsize);
        if (!sd) {
            av_packet_unref(pkt);
            return AVERROR(ENOMEM);
        }
        AV_WL64(sd, (uint64_t)ref);
        avio_seek(pb, roff, SEEK_SET);
        ret = avio_read(pb, sd + 8, rsize);
        if (ret != rsize) {
            av_packet_unref(pkt);
            m->file_pos = -1;
            return AVERROR_EOF;
        }
        m->file_pos = roff + rsize;
    }

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

    // Skip the seek if the previous read left the position here (the same
    // cache as for video).
    if (m->file_pos != m->audio_cursor)
        avio_seek(pb, m->audio_cursor, SEEK_SET);

    // Reads of MIKU_AUDIO_CHUNK_SIZE bytes. The whole FLAC blob is deliberately
    // not returned as one packet: that would undo the interleaving planned in
    // miku_read_header and bring back the queue starvation it avoids on large
    // files.
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

// Finds the largest frame index i whose accumulated pts (the sum of the
// durations of frames [0, i)) is <= target_pts, i.e. the frame that is current
// at that time. pts is not simply the frame index (see next_pts), so this
// scans the index; it works the same in memory and in streaming mode because
// miku_get_index_entry hides the difference. The cost is O(target frame) entry
// reads: a fast in-RAM scan when the index is in memory (bounded by
// MIKU_INDEX_PRELOAD_MAX), but O(n) disk reads from the start of the file in
// streaming mode, as there is no cumulative table to binary-search.
static int64_t miku_frame_for_pts(AVFormatContext *s, int64_t target_pts)
{
    MikuContext *m = s->priv_data;
    int64_t cum = 0;
    int i;

    for (i = 0; i < m->num_frames - 1; i++) {
        int64_t off; int32_t sz, dur, rf; int is_kf;
        int64_t d;
        if (miku_get_index_entry(s, i, &off, &sz, &dur, &is_kf, &rf) < 0)
            break; // best effort -- caller clamps into [0, num_frames)
        d = dur > 0 ? dur : 1;
        if (cum + d > target_pts)
            break;
        cum += d;
    }
    return i;
}

// Sum of the durations of frames [0, upto_idx): the pts that frame upto_idx
// gets from the sequential accumulation in miku_read_video_packet.
// miku_read_seek uses it to re-sync next_pts with wherever current_frame ends
// up, including after the delta-mode keyframe snap, which can land on an
// earlier frame than the one the seek target resolved to.
static int64_t miku_cumulative_duration(AVFormatContext *s, int64_t upto_idx)
{
    MikuContext *m = s->priv_data;
    int64_t cum = 0;
    int i;

    for (i = 0; i < upto_idx && i < m->num_frames; i++) {
        int64_t off; int32_t sz, dur, rf; int is_kf;
        if (miku_get_index_entry(s, i, &off, &sz, &dur, &is_kf, &rf) < 0)
            break; // best effort -- a corrupt tail doesn't need to fail the seek
        cum += dur > 0 ? dur : 1;
    }
    return cum;
}

// Seeks via the index table. Not O(1) in general (see miku_frame_for_pts):
// pts is the running sum of the durations, so a frame index and a pts are not
// interchangeable once durations are non-uniform.
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

        // A delta chain can only be decoded from its root, not from the
        // middle. Walk back to the nearest keyframe, or frame with a
        // long-term reference (just as self-sufficient), at or before
        // frame_idx; sequential reading and the decoder then catch up to the
        // requested frame from there. Non-delta files skip this: every frame
        // is a keyframe, so seeking is exact.
        if (m->delta_mode) {
            int steps = 0;
            for (;;) {
                int64_t off; int32_t sz, dur, rf; int is_kf;
                int ret = miku_get_index_entry(s, video_frame_idx, &off, &sz, &dur, &is_kf, &rf);
                if (ret < 0) return ret;
                // A keyframe is a decodable starting point, and so is a
                // frame with a long-term reference: its packet carries the
                // reference keyframe itself (see miku_read_video_packet), so
                // nothing before it is needed.
                if (is_kf || rf >= 0) break;
                // Bounded by the format's 100-frame chain limit (see mikuenc.c):
                // an unbroken run of non-root frames longer than that is a
                // corrupt index, not a legitimate chain, so fail cleanly
                // instead of walking back through the whole file.
                if (video_frame_idx == 0 || ++steps > 100)
                    return AVERROR_INVALIDDATA;
                video_frame_idx--;
            }
        }
        m->current_frame = video_frame_idx;
        // next_pts is a running counter that miku_read_video_packet only
        // advances sequentially. A seek jumps current_frame directly, so it
        // must re-sync next_pts here, or every later packet would inherit a
        // stale value whichever way, and however far, the seek went.
        m->next_pts = miku_cumulative_duration(s, video_frame_idx);
    }

    // Video seeking snaps back to the nearest preceding keyframe above (a
    // no-op for non-delta files). Audio can only jump to a byte offset
    // estimated in proportion to the playback position, so it is not
    // frame-exact: the FLAC parser resynchronizes on the next valid frame
    // header it finds, and because the proportion is by bytes rather than time
    // the landing point can be far from the requested time when the bitrate
    // varies. The pre-snap frame_idx is used on purpose: audio has no "decode
    // forward from a keyframe and catch up" mechanism the way video does, so
    // the pre-snap position stays closer to what was requested. An accurate
    // seek would need a sample-position -> byte-offset table (a FLAC SEEKTABLE
    // block), which the muxer does not write today.
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
    av_freep(&m->frame_ref);
    return 0;
}

const FFInputFormat ff_miku_demuxer = {
    .p.name         = "miku",
    .p.long_name    = NULL_IF_CONFIG_SMALL("MIKU Lossless Video"),
    .p.extensions   = "miku",
    .p.flags        = AVFMT_NOBINSEARCH | AVFMT_NOGENSEARCH | AVFMT_NO_BYTE_SEEK,
    // Makes libavformat call read_close() when read_header() fails partway
    // through, e.g. after the index arrays are allocated but the index read
    // hits EOF; ASan found a leak there with a header that claimed far more
    // frames than the file holds. read_close() copes with partial
    // initialization, since av_freep() is a no-op on NULL pointers.
    .flags_internal = FF_INFMT_FLAG_INIT_CLEANUP,
    .priv_data_size = sizeof(MikuContext),
    .read_probe     = miku_probe,
    .read_header    = miku_read_header,
    .read_packet    = miku_read_packet,
    .read_seek      = miku_read_seek,
    .read_close     = miku_read_close,
};
