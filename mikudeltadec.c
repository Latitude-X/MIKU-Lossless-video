// mikudeltadec.c — decoder for .miku delta-frame streams (AV_CODEC_ID_MIKU_DELTA)
//
// The first stateful decoder this project has. Every prior version (plain
// WebP keyframes) relied entirely on FFmpeg's stock "webp" decoder, since
// each frame was a standalone, independently decodable image. Delta chains
// break that: reconstructing a delta packet needs real state carried across
// sequential decode calls (the previous frame's true raw pixels), which no
// existing decoder provides.
//
//   - Uses libwebp's simple one-shot WebPDecodeBGRAInto, decoding directly
//     into an already-sized buffer -- this file never touches WebPDecoder/
//     WebPIDecode's incremental API, since every packet here is always a
//     single, complete, self-contained WebP image (either a real keyframe
//     or a delta residual compressed the same way); there's never a reason
//     to decode incrementally across packet boundaries.
//
//   - A packet is a keyframe iff avpkt->flags & AV_PKT_FLAG_KEY, as set by
//     mikudec.c (the demuxer) from the index's keyframe bit. Verified this
//     is safe to rely on despite the encoder-side surprise (see mikuenc.c
//     and AV_PKT_DATA_MIKU_IS_DELTA's comment in packet.h): the mechanism
//     that forces AV_PKT_FLAG_KEY on every ENCODED packet operates on
//     encoder *output* in libavcodec/encode.c and has no decode-side
//     equivalent touching avpkt->flags on the way *into* a decoder -- the
//     analogous decode-side mechanism (libavcodec/decode.c's
//     intra_only_flag) only stamps the OUTPUT AVFrame's flags, and only
//     fires from a codec descriptor's AV_CODEC_PROP_INTRA_ONLY, which
//     AV_CODEC_ID_MIKU_DELTA deliberately has no descriptor entry for at
//     all (adding one marked intra-only would be semantically wrong --
//     delta mode has real inter-frame dependencies -- and would reintroduce
//     this exact class of bug on the output side).
//
//   - Reconstruction is the exact inverse of the encoder's forward delta:
//     encode computed delta[i] = (uint8_t)(curr[i] - ref[i]); decode
//     computes curr[i] = (uint8_t)(ref[i] + delta[i]). Both are plain
//     uint8_t arithmetic (mod 256, i.e. natural wraparound), so this is
//     exact regardless of which individual bytes wrapped during encoding.
//
//   - A keyframe packet is decoded standalone (no reference needed) and
//     immediately becomes the new reference -- this is what makes the
//     demuxer's seek-then-catch-up scheme work: landing on any keyframe is
//     always a valid, self-sufficient place to resume decoding from,
//     independent of whatever reference any other in-flight thread held.
//
//   - FRAME-THREADED (this is the part that changed). The old, purely
//     sequential version kept one plain av_malloc'd ref_frame buffer and
//     copied real pixel bytes into/out of it every call -- correct, but
//     genuinely single-threaded, because a raw byte buffer has no way to
//     let one thread safely wait for another thread's write to finish.
//     This version replaces that with ProgressFrame (progressframe.h),
//     the same mechanism FFmpeg's own reference-frame-dependent
//     frame-threaded decoders use (ffv1dec.c, vp8.c, vp9, pngdec.c,
//     mimic.c) -- verified against a real checkout, and this file's
//     threading is modeled directly on mimic.c's, the smallest of those
//     with a genuinely comparable one-frame-back reference shape. Three
//     pieces work together:
//       1. AV_CODEC_CAP_FRAME_THREADS + FF_CODEC_CAP_USES_PROGRESSFRAMES
//          opt this decoder into the generic round-robin worker pool
//          (pthread_frame.c) instead of the encoder's problem: that pool
//          hands consecutive packets to independent per-thread contexts,
//          exactly the model mikuenc.c's own header comment explains is
//          *unsafe* for reference-dependent work -- unsafe there because
//          encoder frame-threading (frame_thread_encoder.c) has no
//          reference-sharing mechanism at all. Decode-side frame-threading
//          is a different, newer framework and does have one (this file),
//          which is why the two threading models reach opposite
//          conclusions for what looks like the same problem.
//       2. update_thread_context() runs once per frame, strictly in
//          submission order, right as each new thread's context is being
//          prepared -- this is where "the previous frame" actually gets
//          handed from whichever context decoded it to whichever context
//          is about to decode the next one, via ff_progress_frame_ref()
//          (a cheap shared reference, no pixel copy).
//       3. ff_progress_frame_report()/await() are the actual handshake: a
//          delta frame's thread can start its own (slow) WebPDecodeBGRAInto
//          immediately, genuinely in parallel with whatever thread is still
//          finishing the frame it depends on, and only has to actually wait
//          right before the (cheap) per-pixel reconstruction step, which is
//          the one piece of work that truly cannot start early. Reported
//          progress is a single coarse "fully done" per frame, not
//          per-row like mimic.c's -- WebPDecodeBGRAInto is a one-shot call
//          with no partial-progress hook to report through mid-decode, so
//          finer granularity isn't available here the way it is for a
//          decoder with its own row-by-row loop.
//     No AVOptions; DR1 (buffer delegation) and FRAME_THREADS are the only
//     capabilities needed.

#include <limits.h>
#include <string.h>

#include <webp/decode.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "progressframe.h"
#include "thread.h"

typedef struct MikuDeltaDecContext {
    // The immediately preceding frame in this chain, shared (ref-counted,
    // not copied) from whichever thread actually decoded it -- via
    // miku_delta_update_thread_context for a genuinely different thread
    // context, or via the self-update at the end of miku_delta_decode_frame
    // when this same context goes on to decode the very next frame itself
    // (the thread_count == 1 case, and any context's own repeat turns in
    // the round-robin). Blank (prev.f == NULL) until some keyframe has been
    // decoded and propagated this far.
    ProgressFrame prev;
    // This call's own output frame. Becomes *next* frame's `prev` (by
    // whichever mechanism above applies) once allocated -- not only once
    // fully decoded; see ff_thread_finish_setup's placement below for why
    // that's fine (progress tracking, not the reference itself, is what
    // makes waiting for the pixels safe).
    ProgressFrame cur;
} MikuDeltaDecContext;

static av_cold int miku_delta_decode_init(AVCodecContext *avctx)
{
    // Fixed single output format, matching the encoder's own only-accepted
    // input format -- delta reconstruction is only meaningful in the exact
    // pixel layout the encoder computed residuals in. No format-negotiation
    // machinery needed for a single always-used format.
    avctx->pix_fmt = AV_PIX_FMT_BGRA;
    return 0;
}

static av_cold int miku_delta_decode_close(AVCodecContext *avctx)
{
    MikuDeltaDecContext *s = avctx->priv_data;
    ff_progress_frame_unref(&s->cur);
    ff_progress_frame_unref(&s->prev);
    return 0;
}

static void miku_delta_decode_flush(AVCodecContext *avctx)
{
    // Defensive, not load-bearing for correctness, same as the old
    // single-threaded version's own flush: the demuxer's seek logic always
    // lands on a keyframe (see mikudec.c's miku_read_seek), and a keyframe
    // packet never reads `prev` -- it decodes standalone and overwrites
    // `cur` -- so stale references here are self-correcting on the very
    // next packet regardless. Dropped anyway so a decoder in a confused
    // state can't be observed holding frames from before a seek, even
    // transiently, and so those buffers aren't held onto longer than they
    // need to be.
    MikuDeltaDecContext *s = avctx->priv_data;
    ff_progress_frame_unref(&s->cur);
    ff_progress_frame_unref(&s->prev);
}

#if HAVE_THREADS
// Hands "the previous frame" from whichever context just finished setting
// up (src, the most recently dispatched thread) to whichever context is
// about to decode the next packet (dst). Runs once per frame, strictly in
// submission order -- see this file's header comment, point 2. Mirrors
// mimic_decode_update_thread_context (libavcodec/mimic.c) exactly, just
// simplified from that decoder's 16-deep backreference ring down to the
// one-frame-back depth delta mode actually needs.
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

    return 0;
}
#endif

static int miku_delta_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                                    int *got_frame_ptr, AVPacket *avpkt)
{
    MikuDeltaDecContext *s = avctx->priv_data;
    int is_keyframe = (avpkt->flags & AV_PKT_FLAG_KEY) != 0;
    int webp_w, webp_h;
    int ret;

    if (!avpkt->size) {
        *got_frame_ptr = 0;
        return 0;
    }

    if (!is_keyframe && !s->prev.f) {
        // A delta packet with no prior reference means either a corrupt
        // stream or (more likely, if this ever fires) a bug in the
        // demuxer's seek-snapping -- landing anywhere other than exactly a
        // keyframe. Fail clearly rather than reconstruct against a blank
        // reference. Checked before any allocation below, so there's
        // nothing here that needs a matching ff_progress_frame_report --
        // see the header comment on why that matters.
        av_log(avctx, AV_LOG_ERROR,
               "Delta frame with no preceding keyframe in this decode session "
               "(corrupt stream, or seek landed mid-chain)\n");
        return AVERROR_INVALIDDATA;
    }

    if (!WebPGetInfo(avpkt->data, avpkt->size, &webp_w, &webp_h)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid WebP payload in .miku delta packet\n");
        return AVERROR_INVALIDDATA;
    }

    // avctx->width/height are NOT pre-populated by the generic pipeline for
    // this self-describing bitstream format -- confirmed empirically (0x0
    // at decode time on the very first call, same reason the stock webp.c
    // decoder establishes its own dimensions from the bitstream rather than
    // trusting avctx). Establish them from this packet on first use; on
    // every later call, treat a mismatch against the now-established value
    // as the real corruption signal it would be (resolution isn't expected
    // to change mid-stream anywhere in this project's design). Ordinary
    // AVCodecContext fields like these are synced across per-thread
    // contexts by the generic framework itself, independent of this file's
    // own update_thread_context -- the same read/establish/compare pattern
    // that was correct single-threaded stays correct here.
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

    // Release whatever this slot held from a previous lap around the
    // round-robin before allocating fresh -- ff_progress_frame_get_buffer
    // requires a blank ProgressFrame going in, same contract as
    // ff_thread_get_buffer. This also wraps ff_thread_get_buffer itself,
    // so (unlike the old version) there's no separate ff_get_buffer call
    // needed here at all.
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
            // Standalone image: decode straight into this frame's own
            // buffer. Nothing else to do -- unlike the old version, there's
            // no separate ref_frame to also populate; `cur` (shared out via
            // update_thread_context/the self-update below) *is* the
            // reference a following delta frame will read.
            if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, out, out_size, out_stride)) {
                av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (keyframe)\n");
                ret = AVERROR_EXTERNAL;
                goto finish;
            }
        } else {
            // Delta: decode the residual into this frame's own buffer
            // first -- safe and independent, doesn't touch the reference --
            // then wait for the reference frame to be ready, then
            // reconstruct in place: curr = ref + residual (mod 256), the
            // exact inverse of the encoder's curr - ref. The wait is the
            // only part of this branch that can't start immediately; the
            // decode above deliberately happens before it so the expensive
            // work overlaps with whatever thread is finishing the
            // reference instead of being serialized behind it.
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
    // Must run on every path past the allocation above, success or
    // failure: anything already waiting on this frame (a genuinely
    // concurrent thread, since we called ff_thread_finish_setup early)
    // only ever unblocks via a report call. Skipping it on an error return
    // would leave that other thread blocked forever rather than seeing the
    // error the normal way, through this call's own return value. Matches
    // mimic_decode_frame's identical unconditional report before its own
    // error check.
    ff_progress_frame_report(&s->cur, INT_MAX);
    // This context's own `prev`, for the next packet *this exact context*
    // decodes -- relevant when thread_count == 1 (update_thread_context's
    // early return never fires because there's only ever one context) and
    // harmless otherwise (get_thread_context overwrites it before it's
    // read whenever a different context picks up the next frame; see that
    // function's own comment). No-ops cleanly if prev and cur already
    // refer to the same frame.
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
