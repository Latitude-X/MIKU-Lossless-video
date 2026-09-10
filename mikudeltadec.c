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
//     independent of whatever this context's ref_frame held before.
//
//   - No AVOptions, no capabilities beyond DR1: this decoder has real
//     sequential state (ref_frame carried across calls), so it does not
//     declare AV_CODEC_CAP_FRAME_THREADS -- frame-threading's
//     round-robin-across-independent-instances model is exactly as unsafe
//     here as it is for the encoder in delta mode (see mikuenc.c's own
//     capabilities comment for the full reasoning, which applies
//     identically). Unlike the encoder, this decoder has no non-delta mode
//     to preserve parallelism for -- every packet AV_CODEC_ID_MIKU_DELTA
//     ever produces needs this decoder, so there's no tradeoff to weigh;
//     it's simply never frame-parallel.

#include <string.h>

#include <webp/decode.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"

typedef struct MikuDeltaDecContext {
    // Always holds the true, fully-reconstructed raw BGRA pixels of the
    // most recently decoded frame -- exactly mirroring the encoder's own
    // ref_frame (see mikuenc.c), since decode must retrace the same chain
    // encode built. Updated after every successful decode, keyframe or
    // delta alike.
    uint8_t *ref_frame;
    int64_t ref_frame_size;
    int have_reference; // 0 until ref_frame holds a real decoded frame
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
    av_freep(&s->ref_frame);
    return 0;
}

static void miku_delta_decode_flush(AVCodecContext *avctx)
{
    // Defensive, not load-bearing for correctness: the demuxer's seek logic
    // always lands on a keyframe (see mikudec.c's miku_read_seek), and a
    // keyframe packet never reads ref_frame -- it decodes standalone and
    // overwrites it -- so stale state here is self-correcting on the very
    // next packet regardless. Reset anyway so a decoder in a confused state
    // can't be observed holding pixels from before a seek, even transiently.
    MikuDeltaDecContext *s = avctx->priv_data;
    s->have_reference = 0;
}

static int miku_delta_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                                    int *got_frame_ptr, AVPacket *avpkt)
{
    MikuDeltaDecContext *s = avctx->priv_data;
    int is_keyframe = (avpkt->flags & AV_PKT_FLAG_KEY) != 0;
    int webp_w, webp_h;
    int stride;
    int64_t needed;
    int ret;

    if (!avpkt->size) {
        *got_frame_ptr = 0;
        return 0;
    }

    if (!is_keyframe && !s->have_reference) {
        // A delta packet with no prior reference means either a corrupt
        // stream or (more likely, if this ever fires) a bug in the
        // demuxer's seek-snapping -- landing anywhere other than exactly a
        // keyframe. Fail clearly rather than reconstruct against
        // uninitialized/garbage reference data.
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
    // to change mid-stream anywhere in this project's design).
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
    stride = webp_w * 4;
    needed = (int64_t)stride * webp_h;

    if (s->ref_frame_size < needed) {
        // Grown, never shrunk -- same reasoning as the encoder's own
        // scratch/ref_frame buffers (see mikuenc.c): resolution is fixed
        // for the whole stream in practice, so this only actually
        // reallocates once, on the very first call.
        av_freep(&s->ref_frame);
        s->ref_frame = av_malloc(needed);
        if (!s->ref_frame)
            return AVERROR(ENOMEM);
        s->ref_frame_size = needed;
    }

    frame->format = avctx->pix_fmt;
    frame->width  = avctx->width;
    frame->height = avctx->height;
    ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0)
        return ret;

    if (is_keyframe) {
        // Standalone image: decode directly into the reference buffer (this
        // frame's true raw pixels), then copy out to the output frame.
        // Decoding straight into ref_frame and copying out (rather than the
        // reverse) means a delta frame immediately after never needs a
        // separate "which buffer is authoritative" check -- ref_frame is
        // always exactly this frame's pixels once this branch completes.
        if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, s->ref_frame, needed, stride)) {
            av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (keyframe)\n");
            return AVERROR_EXTERNAL;
        }
        av_image_copy_plane(frame->data[0], frame->linesize[0], s->ref_frame, stride,
                             stride, avctx->height);
        s->have_reference = 1;
    } else {
        // Delta: decode the residual into the OUTPUT frame's own buffer
        // first (never directly into ref_frame -- ref_frame must stay the
        // *previous* frame's pixels until reconstruction below has read
        // from it), then reconstruct in place: curr = ref + residual (mod
        // 256), the exact inverse of the encoder's curr - ref. Finally
        // update ref_frame to this frame's newly-reconstructed pixels for
        // whatever comes next in the chain.
        if (!WebPDecodeBGRAInto(avpkt->data, avpkt->size, frame->data[0],
                                 (int64_t)frame->linesize[0] * avctx->height,
                                 frame->linesize[0])) {
            av_log(avctx, AV_LOG_ERROR, "libwebp decode failed (delta residual)\n");
            return AVERROR_EXTERNAL;
        }
        {
            uint8_t *out       = frame->data[0];
            const uint8_t *ref = s->ref_frame;
            int out_stride     = frame->linesize[0];
            for (int y = 0; y < avctx->height; y++) {
                uint8_t *row = out + (int64_t)y * out_stride;
                const uint8_t *ref_row = ref + (int64_t)y * stride;
                for (int x = 0; x < stride; x++)
                    row[x] = (uint8_t)(ref_row[x] + row[x]);
            }
        }
        av_image_copy_plane(s->ref_frame, stride, frame->data[0], frame->linesize[0],
                             stride, avctx->height);
    }

    frame->pts = avpkt->pts;
    frame->flags |= is_keyframe ? AV_FRAME_FLAG_KEY : 0;

    *got_frame_ptr = 1;
    return avpkt->size;
}

const FFCodec ff_miku_delta_decoder = {
    .p.name         = "miku_delta",
    CODEC_LONG_NAME(".miku delta-frame decoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_MIKU_DELTA,
    .p.capabilities = AV_CODEC_CAP_DR1,
    // No AV_CODEC_CAP_FRAME_THREADS -- see this file's own header comment.
    .priv_data_size = sizeof(MikuDeltaDecContext),
    .init           = miku_delta_decode_init,
    .close          = miku_delta_decode_close,
    .flush          = miku_delta_decode_flush,
    FF_CODEC_DECODE_CB(miku_delta_decode_frame),
};
