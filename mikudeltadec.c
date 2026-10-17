// mikudeltadec.c — decoder for delta-chained .miku video (AV_CODEC_ID_MIKU_DELTA)
//
// Selected by the demuxer for codec flags 0x02 and 0x03. Plain WebP streams
// (codec flag 0x01) are independent images and use FFmpeg's stock webp decoder.
// A packet here is a WebP image that is either
//   * a keyframe: a complete picture (AV_PKT_FLAG_KEY, set by mikudec.c), or
//   * a residual to add to a reference picture, byte by byte, modulo 256:
//       - against the previous frame (the default), or
//       - against a long-term reference keyframe, whose own WebP bytes arrive
//         as AV_PKT_DATA_MIKU_REF_FRAME side data (format v3, see below).
//
// Reconstruction is the exact inverse of the encoder's forward delta: the
// encoder stores (uint8_t)(curr[i] - ref[i]) and this decoder computes
// (uint8_t)(ref[i] + delta[i]), so uint8_t wraparound makes it exact. Pixels
// are BGRA, the encoder's only input format. Every packet is one complete WebP
// image, so libwebp's one-shot WebPDecodeBGRAInto is all that is needed (no
// incremental decoding).
//
// Keyframe flag. The decoder trusts AV_PKT_FLAG_KEY on its input packets. The
// flag-forcing that affects the encoder (see AV_PKT_DATA_MIKU_IS_DELTA in
// packet.h) has no decode-side counterpart for packet flags: libavcodec's
// decode.c only stamps AV_FRAME_FLAG_KEY on OUTPUT frames, and only for codecs
// whose descriptor has AV_CODEC_PROP_INTRA_ONLY. The AV_CODEC_ID_MIKU_DELTA
// descriptor in codec_desc.c (miku-support.patch) deliberately leaves that
// property off, because delta frames do depend on earlier data; the descriptor
// must still exist, since ff_decode_preinit dereferences it.
//
// Chain roots. A keyframe and a frame with a long-term reference both decode
// without the previous frame, so decoding can start at either. That is what
// lets the demuxer's seek (mikudec.c) land on either kind and catch up from
// there.
//
// Frame threading (AV_CODEC_CAP_FRAME_THREADS + FF_CODEC_CAP_USES_PROGRESSFRAMES),
// modelled on mimic.c, the smallest in-tree decoder whose frames reference the
// preceding frame through ProgressFrame:
//   1. The generic frame-thread pool (pthread_frame.c) hands consecutive
//      packets to separate codec contexts.
//   2. update_thread_context() runs once per frame, in submission order, and
//      passes the previous frame from the context that started it to the one
//      about to decode the next packet, as a shared ProgressFrame reference
//      (no pixel copy).
//   3. ff_progress_frame_report()/await() synchronise the pixels. A delta frame
//      starts its slow WebPDecodeBGRAInto at once, in parallel with the thread
//      still finishing the frame it depends on, and waits only right before
//      the cheap per-pixel add. Progress is reported once, as "complete":
//      WebPDecodeBGRAInto is a one-shot call with no partial progress to
//      report per row.
//
// Long-term references (format v3). A delta packet may carry
// AV_PKT_DATA_MIKU_REF_FRAME side data: the reference keyframe's frame number
// followed by that keyframe's WebP bytes (attached by mikudec.c). Such a frame
// is a chain root: it never waits on `prev` and decodes fully in parallel with
// its neighbours. The reference keyframe has to be decoded to be used, and runs
// of frames often share one (a returning scene, a periodic re-root of a static
// shot), so decoded references live in a small cache keyed by the keyframe's
// compressed bytes. Frame numbers are only meaningful within one file and a
// decoder can be fed several (concat, playlists), so they are not used as keys.
// The cache is shared by every frame-thread context through
// update_thread_context (one locked object behind an AVBufferRef); entries are
// immutable once inserted. Two threads that miss on the same keyframe at the
// same moment both decode it: duplicated work, never a wrong result.

#include <inttypes.h>
#include <limits.h>
#include <string.h>

#include <webp/decode.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "progressframe.h"
#include "thread.h"
#include "libavutil/buffer.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "libavutil/thread.h"

// Decoded long-term reference keyframes kept ready for reuse. A few is
// plenty: playback normally leans on one or two references at a time.
#define MIKU_REF_CACHE_N 4

// Entries are matched by the keyframe's compressed BYTES, not by the frame
// number the demuxer reports: that number is only meaningful inside one file,
// and a decoder can be fed several (the concat demuxer, a playlist), where two
// different files both have a "frame 0". Comparing the bytes is exact, and
// costs a memcmp per long-term-reference frame -- these are chain roots, a
// small fraction of all frames, and the compare is far cheaper than the decode
// it saves.
typedef struct MikuRefEntry {
    AVBufferRef *bytes; // the compressed keyframe this entry was decoded from
    AVBufferRef *buf;   // decoded BGRA, tightly packed (stride == width * 4)
    uint64_t     stamp; // LRU clock value at last use
} MikuRefEntry;

typedef struct MikuRefCache {
    AVMutex      lock;  // guards ent[] and clock; the pixel data itself is
                        // immutable once an entry is inserted
    MikuRefEntry ent[MIKU_REF_CACHE_N];
    uint64_t     clock;
} MikuRefCache;

typedef struct MikuDeltaDecContext {
    // The preceding frame, shared by reference (no pixel copy). Set by
    // miku_delta_update_thread_context for a different thread context, or by
    // ff_progress_frame_replace at the end of miku_delta_decode_frame when
    // this same context decodes the next frame (thread_count == 1). NULL until
    // a chain root has been decoded.
    ProgressFrame prev;
    // This call's output frame. It becomes the next frame's `prev` as soon as
    // it is allocated (ff_thread_finish_setup), before it is fully decoded;
    // the next frame's ff_progress_frame_await is what waits for the pixels.
    ProgressFrame cur;
    // Long-term reference cache, shared by all frame-thread contexts (see
    // the header comment). Never NULL after a successful init.
    AVBufferRef *cache_ref;
} MikuDeltaDecContext;

static void miku_ref_cache_free(void *opaque, uint8_t *data)
{
    MikuRefCache *c = (MikuRefCache *)data;
    for (int i = 0; i < MIKU_REF_CACHE_N; i++) {
        av_buffer_unref(&c->ent[i].buf);
        av_buffer_unref(&c->ent[i].bytes);
    }
    ff_mutex_destroy(&c->lock);
    av_free(c);
}

static AVBufferRef *miku_ref_cache_new(void)
{
    MikuRefCache *c = av_mallocz(sizeof(*c));
    AVBufferRef *ref;

    if (!c)
        return NULL;
    if (ff_mutex_init(&c->lock, NULL)) {
        av_free(c);
        return NULL;
    }
    ref = av_buffer_create((uint8_t *)c, sizeof(*c), miku_ref_cache_free, NULL, 0);
    if (!ref) {
        ff_mutex_destroy(&c->lock);
        av_free(c);
        return NULL;
    }
    return ref;
}

// Returns (through *out, a new reference the caller must unref) the decoded
// pixels of the long-term reference keyframe whose compressed bytes are
// data[0..size), decoding it into the cache first if it isn't there yet.
// Pixels are tightly packed BGRA. `id` (the frame number) is for messages only.
static int miku_ref_get(AVCodecContext *avctx, AVBufferRef *cache_ref, int64_t id,
                        const uint8_t *data, size_t size, AVBufferRef **out)
{
    MikuRefCache *c = (MikuRefCache *)cache_ref->data;
    int64_t need = (int64_t)avctx->width * avctx->height * 4;
    // Decoded keyframes are full frames; keep the cache's footprint bounded
    // for very large pictures (the minimum is always one entry).
    int slots = FFMAX(1, FFMIN(MIKU_REF_CACHE_N, (int)((256LL << 20) / FFMAX(need, 1))));
    AVBufferRef *buf, *bytes;
    int w, h, victim = 0;

    ff_mutex_lock(&c->lock);
    for (int i = 0; i < slots; i++) {
        if (c->ent[i].buf && c->ent[i].bytes->size == size &&
            !memcmp(c->ent[i].bytes->data, data, size)) {
            c->ent[i].stamp = ++c->clock;
            *out = av_buffer_ref(c->ent[i].buf);
            ff_mutex_unlock(&c->lock);
            return *out ? 0 : AVERROR(ENOMEM);
        }
    }
    ff_mutex_unlock(&c->lock);

    // Miss: decode it (outside the lock -- this is the slow part).
    if (size > INT_MAX || !WebPGetInfo(data, size, &w, &h) ||
        w != avctx->width || h != avctx->height) {
        av_log(avctx, AV_LOG_ERROR,
               "Invalid long-term reference keyframe (frame %"PRId64")\n", id);
        return AVERROR_INVALIDDATA;
    }
    buf = av_buffer_alloc(need);
    if (!buf)
        return AVERROR(ENOMEM);
    if (!WebPDecodeBGRAInto(data, size, buf->data, need, avctx->width * 4)) {
        av_log(avctx, AV_LOG_ERROR,
               "libwebp decode failed (long-term reference, frame %"PRId64")\n", id);
        av_buffer_unref(&buf);
        return AVERROR_EXTERNAL;
    }

    // Keep a private copy of the compressed bytes as the cache key (the
    // packet's side data goes away with the packet). If either allocation
    // fails the frame is still decoded correctly; it just isn't cached.
    bytes = av_buffer_alloc(size);
    if (bytes)
        memcpy(bytes->data, data, size);

    ff_mutex_lock(&c->lock);
    for (int i = 0; i < slots; i++) {
        if (c->ent[i].buf && c->ent[i].bytes->size == size &&
            !memcmp(c->ent[i].bytes->data, data, size)) {
            // Another thread decoded the same keyframe meanwhile; keep
            // theirs in the cache and just use ours for this one frame.
            c->ent[i].stamp = ++c->clock;
            ff_mutex_unlock(&c->lock);
            av_buffer_unref(&bytes);
            *out = buf;
            return 0;
        }
    }
    for (int i = 0; i < slots; i++) {
        if (!c->ent[i].buf) { victim = i; break; }
        if (c->ent[i].stamp < c->ent[victim].stamp)
            victim = i;
    }
    if (bytes) {
        AVBufferRef *keep = av_buffer_ref(buf); // NULL just means: don't cache
        if (keep) {
            // Readers hold their own references, so dropping the victim's is safe.
            av_buffer_unref(&c->ent[victim].buf);
            av_buffer_unref(&c->ent[victim].bytes);
            c->ent[victim].buf   = keep;
            c->ent[victim].bytes = bytes;
            c->ent[victim].stamp = ++c->clock;
            bytes = NULL;
        }
    }
    ff_mutex_unlock(&c->lock);
    av_buffer_unref(&bytes); // still ours only if it was not cached
    *out = buf;
    return 0;
}

static av_cold int miku_delta_decode_init(AVCodecContext *avctx)
{
    // Output is always BGRA, the layout the encoder computed its residuals in.
    MikuDeltaDecContext *s = avctx->priv_data;

    avctx->pix_fmt = AV_PIX_FMT_BGRA;

    s->cache_ref = miku_ref_cache_new();
    if (!s->cache_ref)
        return AVERROR(ENOMEM);
    return 0;
}

static av_cold int miku_delta_decode_close(AVCodecContext *avctx)
{
    MikuDeltaDecContext *s = avctx->priv_data;
    ff_progress_frame_unref(&s->cur);
    ff_progress_frame_unref(&s->prev);
    av_buffer_unref(&s->cache_ref);
    return 0;
}

static void miku_delta_decode_flush(AVCodecContext *avctx)
{
    // Not needed for correctness: the demuxer's seek always lands on a chain
    // root (a keyframe or a frame with a long-term reference) and neither
    // reads `prev`. Dropping the references anyway means no frames from before
    // the seek stay alive. The reference cache is kept: its entries are
    // immutable and keyed by content, so they stay valid across a seek.
    MikuDeltaDecContext *s = avctx->priv_data;
    ff_progress_frame_unref(&s->cur);
    ff_progress_frame_unref(&s->prev);
}

#if HAVE_THREADS
// Passes "the previous frame" and the shared reference cache from the context
// that just finished setting up (src, the most recently dispatched thread) to
// the context about to decode the next packet (dst). Runs once per frame, in
// submission order. Patterned on mimic_decode_update_thread_context
// (libavcodec/mimic.c), reduced to the one-frame-back depth used here.
static int miku_delta_update_thread_context(AVCodecContext *avctx,
                                             const AVCodecContext *avctx_from)
{
    MikuDeltaDecContext *dst = avctx->priv_data;
    const MikuDeltaDecContext *src = avctx_from->priv_data;

    if (avctx == avctx_from)
        return 0; // same context (e.g. thread_count == 1); nothing to hand off

    ff_progress_frame_unref(&dst->prev);
    if (src->cur.f)
        ff_progress_frame_ref(&dst->prev, &src->cur);

    // Converge every thread context on one shared reference cache. Each
    // context starts with a private one from init(); after the first lap of
    // the round-robin they all point at the same object, so a reference
    // keyframe is decoded once for all threads, not once per thread.
    if (src->cache_ref && dst->cache_ref != src->cache_ref) {
        av_buffer_unref(&dst->cache_ref);
        dst->cache_ref = av_buffer_ref(src->cache_ref);
        if (!dst->cache_ref)
            return AVERROR(ENOMEM);
    }

    return 0;
}
#endif

static int miku_delta_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                                    int *got_frame_ptr, AVPacket *avpkt)
{
    MikuDeltaDecContext *s = avctx->priv_data;
    int is_keyframe = (avpkt->flags & AV_PKT_FLAG_KEY) != 0;
    size_t ref_sd_size = 0;
    const uint8_t *ref_sd = av_packet_get_side_data(avpkt, AV_PKT_DATA_MIKU_REF_FRAME,
                                                     &ref_sd_size);
    // A delta frame with a long-term reference: self-contained, a chain root.
    int has_ref = !is_keyframe && ref_sd && ref_sd_size > 8;
    AVBufferRef *refbuf = NULL;
    int webp_w, webp_h;
    int ret;

    if (!avpkt->size) {
        *got_frame_ptr = 0;
        return 0;
    }

    if (!is_keyframe && !has_ref && !s->prev.f) {
        // A residual against the previous frame, but there is none: a corrupt
        // stream, or a demuxer seek that landed on something other than a
        // keyframe or a frame with a long-term reference. Fail rather than
        // reconstruct against a blank picture. Nothing has been allocated
        // yet, so there is no frame to report progress on (see `finish:`).
        av_log(avctx, AV_LOG_ERROR,
               "Delta frame with no preceding keyframe in this decode session "
               "(corrupt stream, or seek landed mid-chain)\n");
        return AVERROR_INVALIDDATA;
    }

    if (!WebPGetInfo(avpkt->data, avpkt->size, &webp_w, &webp_h)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid WebP payload in .miku delta packet\n");
        return AVERROR_INVALIDDATA;
    }

    // The demuxer leaves width/height unset (WebP is self-describing, and the
    // stock webp decoder also takes them from the bitstream), so the first
    // packet establishes them. A later mismatch is treated as corruption: the
    // format has no mid-stream resolution change. Frame-thread contexts get
    // these AVCodecContext fields from the generic framework, not from
    // update_thread_context below.
    if (!avctx->width || !avctx->height) {
        ret = ff_set_dimensions(avctx, webp_w, webp_h);
        if (ret < 0)
            return ret;
    } else if (webp_w != avctx->width || webp_h != avctx->height) {
        av_log(avctx, AV_LOG_ERROR,
               "Payload dimensions %dx%d don't match stream dimensions %dx%d\n",
               webp_w, webp_h, avctx->width, avctx->height);
        return AVERROR_INVALIDDATA;
    }

    // Drop whatever this context held from its previous turn:
    // ff_progress_frame_get_buffer requires a blank ProgressFrame (it wraps
    // ff_thread_get_buffer, so no separate ff_get_buffer call is needed).
    ff_progress_frame_unref(&s->cur);
    ret = ff_progress_frame_get_buffer(avctx, &s->cur, AV_GET_BUFFER_FLAG_REF);
    if (ret < 0)
        return ret;

    // Everything a later thread's update_thread_context needs from this
    // call (s->cur itself, now allocated) is already fixed. Hand off here,
    // before doing any of the actual (slow) pixel work below, instead of
    // only implicitly when this function returns -- this is what lets the
    // next frame's WebPDecodeBGRAInto genuinely overlap with this one's
    // rather than waiting for this whole call to finish first. Past this
    // point: don't reallocate s->cur, only write through the buffer it
    // already points at (per ff_thread_finish_setup's own contract).
    ff_thread_finish_setup(avctx);

    {
        uint8_t *out        = s->cur.f->data[0];
        int      out_stride = s->cur.f->linesize[0];
        int64_t  out_size   = (int64_t)out_stride * avctx->height;

        if (is_keyframe) {
            // Standalone image: decode straight into this frame's buffer.
            // `cur` is shared onward (update_thread_context, or the
            // ff_progress_frame_replace below) and is what the next delta
            // frame reads as its reference.
            if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, out, out_size, out_stride)) {
                av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (keyframe)\n");
                ret = AVERROR_EXTERNAL;
                goto finish;
            }
        } else if (has_ref) {
            // Delta against a long-term reference keyframe. Everything this
            // needs is in the packet itself (see mikudec.c), so there is no
            // wait on `prev` and no ordering against other threads at all.
            ret = miku_ref_get(avctx, s->cache_ref, (int64_t)AV_RL64(ref_sd),
                               ref_sd + 8, ref_sd_size - 8, &refbuf);
            if (ret < 0)
                goto finish;
            if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, out, out_size, out_stride)) {
                av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (delta residual, long-term ref)\n");
                ret = AVERROR_EXTERNAL;
                goto finish;
            }
            {
                const uint8_t *ref = refbuf->data;
                int ref_stride = avctx->width * 4; // tightly packed, see miku_ref_get
                for (int y = 0; y < avctx->height; y++) {
                    uint8_t       *row     = out + (int64_t)y * out_stride;
                    const uint8_t *ref_row = ref + (int64_t)y * ref_stride;
                    for (int x = 0; x < webp_w * 4; x++)
                        row[x] = (uint8_t)(ref_row[x] + row[x]);
                }
            }
        } else {
            // Residual against the previous frame. Decode it into this
            // frame's own buffer first (it does not depend on the reference,
            // so it overlaps with the thread still finishing that frame), then
            // wait for the reference and add: curr = ref + residual (mod 256).
            if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, out, out_size, out_stride)) {
                av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (delta residual)\n");
                ret = AVERROR_EXTERNAL;
                goto finish;
            }

            ff_progress_frame_await(&s->prev, INT_MAX);
            {
                const uint8_t *ref        = s->prev.f->data[0];
                int             ref_stride = s->prev.f->linesize[0];
                for (int y = 0; y < avctx->height; y++) {
                    uint8_t       *row     = out + (int64_t)y * out_stride;
                    const uint8_t *ref_row = ref + (int64_t)y * ref_stride;
                    for (int x = 0; x < webp_w * 4; x++)
                        row[x] = (uint8_t)(ref_row[x] + row[x]);
                }
            }
        }
    }

    s->cur.f->pict_type = is_keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;

    ret = av_frame_ref(frame, s->cur.f);
    if (ret < 0)
        goto finish;

    frame->pts = avpkt->pts;
    frame->flags |= is_keyframe ? AV_FRAME_FLAG_KEY : 0;
    *got_frame_ptr = 1;
    ret = avpkt->size;

finish:
    av_buffer_unref(&refbuf);
    // Must run on every path past the allocation above, success or
    // failure: anything already waiting on this frame (a genuinely
    // concurrent thread, since we called ff_thread_finish_setup early)
    // only ever unblocks via a report call. Skipping it on an error return
    // would leave that other thread blocked forever rather than seeing the
    // error the normal way, through this call's own return value. Matches
    // mimic_decode_frame's identical unconditional report before its own
    // error check.
    ff_progress_frame_report(&s->cur, INT_MAX);
    // Point this context's own `prev` at the frame just decoded, for the next
    // packet this same context handles. That matters when thread_count == 1
    // (update_thread_context returns early for a single context); otherwise
    // update_thread_context overwrites `prev` before it is read.
    ff_progress_frame_replace(&s->prev, &s->cur);

    if (ret < 0 && !(avctx->active_thread_type & FF_THREAD_FRAME))
        ff_progress_frame_unref(&s->cur); // single-threaded: no generic cleanup path watches this

    return ret;
}

const FFCodec ff_miku_delta_decoder = {
    .p.name         = "miku_delta",
    CODEC_LONG_NAME(".miku delta-frame decoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_MIKU_DELTA,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_FRAME_THREADS,
    UPDATE_THREAD_CONTEXT(miku_delta_update_thread_context),
    .caps_internal  = FF_CODEC_CAP_USES_PROGRESSFRAMES | FF_CODEC_CAP_INIT_CLEANUP,
    .priv_data_size = sizeof(MikuDeltaDecContext),
    .init           = miku_delta_decode_init,
    .close          = miku_delta_decode_close,
    .flush          = miku_delta_decode_flush,
    FF_CODEC_DECODE_CB(miku_delta_decode_frame),
};
