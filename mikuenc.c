// mikuenc.c — fast, frame-parallel lossless WebP (VP8L) encoder for .miku,
// with delta-frame support
//
// Design summary (see the chat reply for the full plan and the benchmarks
// that motivated it):
//
//   - Conversion isn't the bottleneck. Measured on this machine: YUV420P->
//     BGRA via swscale costs ~1.7ms/frame at 1080p regardless of content or
//     settings. VP8L's own compression search costs 9ms-2900ms/frame
//     depending on content complexity and effort level -- three orders of
//     magnitude more variation, and it's where all the real time goes.
//     So this encoder does NOT do its own pixel-format conversion at all:
//     it declares AV_PIX_FMT_BGRA as its only accepted input format and
//     lets FFmpeg's ordinary implicit-scale pipeline (backed by swscale,
//     already well-optimized) handle anything else upstream.
//
//   - THREADING HAS CHANGED from the original dedup-only version. That
//     version got its parallelism for free from AV_CODEC_CAP_FRAME_THREADS
//     (the generic framework opening N independent encoder instances and
//     round-robining frames across them) because every frame really was
//     independent. Delta chains break that: consecutive frames need shared
//     reference-frame state, which round-robin-across-independent-instances
//     cannot provide, and there is no reliable way for one instance to
//     detect it's being opened as a frame-threading worker and refuse (see
//     the reply, and the git history of this file, for the full trace
//     through frame_thread_encoder.c that established this).
//
//     This version replaces that entirely with its own hand-rolled
//     producer/worker-pool architecture, used uniformly for BOTH
//     delta_frames=0 and delta_frames=1 -- not just to make delta mode
//     safe, but because it restores real multi-core parallelism for
//     delta_frames=0 too, which AV_CODEC_CAP_FRAME_THREADS's removal had
//     otherwise cost. AV_CODEC_CAP_FRAME_THREADS is no longer declared at
//     all; this encoder now always manages its own threads.
//
//     The key insight the architecture rests on: only *delta computation*
//     needs to be sequential (cheap, per-pixel subtract, plus the adaptive
//     block-ending heuristic) -- it runs on the calling thread, in the
//     encode() callback itself, exactly as before, so it's automatically
//     ordered correctly (only one thread ever calls encode()). What moves
//     onto worker threads is only the expensive part: WebPEncode itself,
//     which has no cross-frame dependency regardless of mode. A small pool
//     of persistent worker threads pulls pre-computed pixel buffers
//     (either a raw keyframe or an already-computed delta residual --
//     worker threads never touch ref_frame or do any subtraction) from a
//     ring buffer and compresses them in parallel; the main thread hands
//     packets back to the caller in strict submission order regardless of
//     which worker finishes which job when, via the exact same
//     finished_task_index bookkeeping frame_thread_encoder.c uses for the
//     analogous problem (see miku_encode_frame's retrieval logic, and
//     MIKU_TASK_SLACK's comment, for why that specific pattern -- adopted
//     directly rather than re-derived).
//
//     One correctness point that only matters once multiple WebPEncode
//     calls can be in flight at once (never an issue in the old, purely
//     sequential version): the delta-buffer exact=1 override (see below)
//     used to be implemented by flipping s->config.exact and restoring it
//     right after a single synchronous WebPEncode call. With N workers
//     potentially calling WebPEncode concurrently, mutating a *shared*
//     config like that is a real data race. Each worker now builds its own
//     local WebPConfig copy (a cheap struct copy; s->config itself is never
//     mutated after init(), so concurrent reads of it are safe) and
//     overrides .exact on that local copy only, per task.
//
//   - It's a thin, correctness-preserving wrapper around libwebp's actual
//     VP8L implementation (WebPPicture / WebPConfig / WebPEncode), not a
//     reimplementation. See the reply for why a from-scratch or GPU VP8L
//     encoder is not a short-term-feasible alternative to this.
//
//   - Every frame is copied into an owned buffer before being handed to
//     WebPEncode, even though BGRA already matches WebP's ARGB memory
//     layout byte-for-byte and a direct pointer into frame->data[0] would
//     look like a free zero-copy optimization. It isn't safe: tested
//     empirically, WebPEncode mutates its argb input in place (with
//     exact=0, it zeroes RGB under fully-transparent pixels in the actual
//     buffer it was given, not a private copy). Pointing it straight at
//     the caller's AVFrame would silently corrupt that frame.
//
//   - Delta buffers carry a residual in their alpha channel too (curr_alpha
//     - ref_alpha), not real transparency -- but WebPEncode doesn't know
//     that. Its default exact=0 behavior (correct and desired for a real
//     image's genuinely-transparent pixels) actively corrupts a delta
//     buffer at every pixel whose real alpha simply didn't change between
//     frames (residual alpha 0 there is common, not rare): confirmed
//     empirically, an all-zero-alpha-delta frame (real content, alpha just
//     constant) decoded back as an entirely blank image, RGB deltas
//     silently discarded. Forcing exact=1 specifically for delta-buffer
//     tasks (see above) fixes this without changing the user's own -exact
//     setting for keyframes.
//
//   - HDR/color metadata needs no special-case code here at all: it's
//     handled generically by libavcodec/encode.c (color_primaries/trc/
//     space/range on AVCodecContext, and coded_side_data for mastering-
//     display/content-light) before/after this encoder's own callback
//     runs, identical to how it already works for every other encoder.
//
//   - WebPConfig.thread_level ("try internal multi-threaded encoding") is
//     exposed as an option but its benefit is unverified here (see the
//     reply for the benchmark); it's a completely separate axis from the
//     worker-pool threading in this file -- one parallelizes the search
//     *within* a single image, the other parallelizes *across* images.

#include <stdatomic.h>
#include <string.h>

#include <webp/encode.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/internal.h"
#include "libavutil/thread.h"
#include "pthread_internal.h"

// One extra slot beyond the worker count, same rationale as
// frame_thread_encoder.c's own BUFFER_SIZE (MAX_THREADS + 2): with a ring
// buffer sized to exactly the thread count, the "queue full" and "queue
// empty" states become indistinguishable (task_index could wrap around to
// equal finished_task_index in both cases). One spare slot resolves the
// ambiguity. Adopted directly from that proven, real, in-tree pattern
// rather than re-derived.
#define MIKU_TASK_SLACK 2

typedef struct MikuTask {
    // Input, set by the main thread before dispatch. pixels is fixed-size
    // (allocated once in init(), sized to the stream's resolution) and
    // never reallocated -- avoids any chance of a worker reading a buffer
    // the main thread is mid-realloc on. Exclusively owned by whichever
    // side currently "has" this slot: the main thread while filling it
    // before dispatch and after retrieval, a single worker thread while
    // task_pending is set and finished is not yet set. The ring-buffer
    // backpressure logic (mirroring frame_thread_encoder.c) guarantees
    // the main thread never reuses a slot before the previous occupant's
    // packet has been retrieved, so there's never a moment two threads
    // both believe they own the same slot.
    uint8_t *pixels;
    int is_keyframe; // decided by the sequential producer step, before
                      // dispatch -- workers never compute this themselves
    int64_t pts;
    int64_t duration; // carried through explicitly: AV_CODEC_CAP_DELAY
                       // (see the FFCodec struct below) means the generic
                       // pipeline no longer derives pkt duration for us
                       // from the input frame the way it used to.

    // Output, set by the worker, read by the main thread at retrieval.
    uint8_t *out_data; // owned; freed via WebPMemoryWriterClear at
                        // retrieval (or at close(), for any task still
                        // outstanding -- shouldn't happen after a full
                        // flush, but close() doesn't assume flush ran)
    size_t out_size;
    int encode_failed;
    int webp_error_code;

    int finished; // guarded by finished_task_mutex
} MikuTask;

typedef struct MikuEncContext {
    AVClass *class;
    WebPConfig config; // never mutated after init() -- safe for concurrent
                        // reads from multiple worker threads
    int method; // 0 (fastest) - 6 (slowest/best); WebPConfig's own documented range
    int exact;  // preserve exact RGB under fully-transparent alpha (matches WebPConfig.exact)
    int thread_level; // maps to WebPConfig.thread_level -- see the reply for why this
                       // is exposed despite unproven benefit on this specific hardware

    int delta_frames; // -delta_frames option; see miku_encode_init for what
                       // flipping this does to avctx->codec_id
    double delta_threshold; // -delta_threshold: fraction of unchanged bytes
                       // (0..1) above which a candidate frame is encoded as
                       // a delta rather than ending the block early. NOT YET
                       // empirically tuned against real footage the way
                       // method/thread_level were -- 0.5 is a reasonable
                       // starting point, exposed as an option specifically
                       // so it can be benchmarked and adjusted later rather
                       // than hardcoded.
    int delta_method;  // -delta_method: WebP method (effort level) for delta-
                        // residual tasks specifically, independent of -method
                        // (keyframes still use that). See miku_worker and the
                        // reply: the frames_in_block/delta_threshold heuristic
                        // above is a single whole-frame statistic, so a small
                        // persistently-complex pocket inside an otherwise-
                        // static frame (noise, heavy grain) can clear
                        // delta_threshold easily while still costing a full
                        // keyframe's worth of VP8L search time. This bounds
                        // that worst case directly rather than trying to
                        // predict it upfront.

    // ---- Sequential producer state (touched only by the thread calling
    // encode(), which is always exactly one thread -- none of this needs
    // locking, same as it never did before threading existed here) ----

    // scratch always holds this call's frame's true raw pixels, copied
    // fresh from the AVFrame every call. Reused across calls (grown, never
    // shrunk) to avoid a malloc/free per frame.
    uint8_t *scratch;
    int64_t scratch_size;

    // ref_frame always holds a clean copy of the most recently processed
    // frame's RAW pixels -- "frame N-1" while processing frame N.
    // Deliberately a separate buffer from `scratch`, updated by explicit
    // copy before dispatch: whichever task slot ends up holding frame N's
    // pixels is about to be handed to a worker that may run at any time
    // after dispatch, so ref_frame must already have its own independent
    // copy before that happens, not rely on reading back from a slot a
    // worker might be touching.
    uint8_t *ref_frame;
    int64_t ref_frame_size;
    int have_reference; // 0 until ref_frame holds a real previous frame

    uint8_t *delta_buf; // scratch for the residual (curr - ref, mod 256);
                         // same size as scratch/ref_frame once allocated
    int64_t delta_buf_size;

    int frames_in_block; // frames encoded since (and including) the current
                          // block's keyframe; block ends (forcing a fresh
                          // keyframe) at 100 regardless of how favorable
                          // deltas remain, per spec

    // ---- Worker pool ----
    int nthreads;
    pthread_t *workers;
    MikuTask *tasks;
    unsigned ntasks; // nthreads + MIKU_TASK_SLACK

    pthread_mutex_t task_fifo_mutex; // guards next_task_index/task_index
    pthread_cond_t  task_fifo_cond;
    pthread_mutex_t finished_task_mutex; // guards tasks[i].finished
    pthread_cond_t  finished_task_cond;
    unsigned pthread_init_cnt; // required by ff_pthread_init/free, see
                               // pthread_internal.h

    unsigned next_task_index;     // next slot a worker will claim
    unsigned task_index;          // next slot the main thread will fill
    unsigned finished_task_index; // oldest slot not yet retrieved

    atomic_int exit;
    int workers_started; // how many pthread_create calls actually
                          // succeeded, so close() joins exactly those
} MikuEncContext;

#define OFF(member) offsetof(MikuEncContext, member)
DEFINE_OFFSET_ARRAY(MikuEncContext, miku_enc_ctx, pthread_init_cnt,
                    (OFF(task_fifo_mutex), OFF(finished_task_mutex)),
                    (OFF(task_fifo_cond),  OFF(finished_task_cond)));
#undef OFF

static void *miku_worker(void *arg)
{
    AVCodecContext *avctx = arg;
    MikuEncContext *s = avctx->priv_data;

    for (;;) {
        MikuTask *task;
        unsigned idx;
        WebPConfig local_config;
        WebPPicture pic;
        WebPMemoryWriter writer;

        pthread_mutex_lock(&s->task_fifo_mutex);
        while (s->next_task_index == s->task_index || atomic_load(&s->exit)) {
            if (atomic_load(&s->exit)) {
                pthread_mutex_unlock(&s->task_fifo_mutex);
                return NULL;
            }
            pthread_cond_wait(&s->task_fifo_cond, &s->task_fifo_mutex);
        }
        idx = s->next_task_index;
        s->next_task_index = (s->next_task_index + 1) % s->ntasks;
        pthread_mutex_unlock(&s->task_fifo_mutex);

        task = &s->tasks[idx];

        // Local config copy: see this file's header comment for why this
        // can't be a mutate-then-restore on the shared s->config anymore
        // now that multiple workers can be inside WebPEncode at once.
        local_config = s->config;
        if (task->is_keyframe == 0) {
            local_config.exact = 1;
            // Independent of exact: retrieval in miku_encode_frame is
            // strict submission order, so one task stuck in a slow
            // WebPEncode call blocks every later packet no matter how
            // many other workers are idle -- see the reply for why that
            // turns a single locally-complex frame into a sustained
            // playback slowdown rather than one skipped beat.
            if (local_config.method > s->delta_method)
                local_config.method = s->delta_method;
        }

        if (!WebPPictureInit(&pic)) {
            task->encode_failed = 1;
            task->webp_error_code = -1;
        } else {
            pic.width    = avctx->width;
            pic.height   = avctx->height;
            pic.use_argb = 1;
            pic.argb     = (uint32_t *)task->pixels;
            pic.argb_stride = avctx->width;

            WebPMemoryWriterInit(&writer);
            pic.writer     = WebPMemoryWrite;
            pic.custom_ptr = &writer;

            if (!WebPEncode(&local_config, &pic)) {
                task->encode_failed    = 1;
                task->webp_error_code  = pic.error_code;
                WebPMemoryWriterClear(&writer);
            } else {
                task->encode_failed = 0;
                task->out_data = writer.mem;
                task->out_size = writer.size;
            }
        }

        pthread_mutex_lock(&s->finished_task_mutex);
        task->finished = 1;
        pthread_cond_signal(&s->finished_task_cond);
        pthread_mutex_unlock(&s->finished_task_mutex);
    }
}

static av_cold int miku_encode_init(AVCodecContext *avctx)
{
    MikuEncContext *s = avctx->priv_data;
    int ret;

    if (avctx->pix_fmt != AV_PIX_FMT_BGRA) {
        // Shouldn't happen -- FFmpeg only hands us formats from p.pix_fmts --
        // but WebPEncode's correctness depends entirely on argb/argb_stride
        // actually being BGRA-laid-out memory, so this is cheap insurance
        // against ever silently encoding garbage.
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format (expected bgra)\n");
        return AVERROR(EINVAL);
    }

    // Flip the output codec_id before any downstream code (the muxer, via
    // st->codecpar->codec_id) reads it, so a delta-mode stream is never
    // mistaken for a plain WebP one. Verified empirically end-to-end that
    // this propagates correctly.
    if (s->delta_frames)
        avctx->codec_id = AV_CODEC_ID_MIKU_DELTA;

    if (!WebPConfigInit(&s->config)) {
        av_log(avctx, AV_LOG_ERROR,
               "WebPConfigInit failed (libwebp header/library ABI mismatch)\n");
        return AVERROR_EXTERNAL;
    }
    if (!WebPConfigLosslessPreset(&s->config, s->method)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid method %d\n", s->method);
        return AVERROR(EINVAL);
    }
    s->config.lossless = 1;
    s->config.quality  = 100;
    s->config.exact    = s->exact;
    s->config.thread_level = s->thread_level;

    if (!WebPValidateConfig(&s->config)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid libwebp lossless configuration\n");
        return AVERROR(EINVAL);
    }

    {
        int64_t needed = (int64_t)avctx->width * 4 * avctx->height;
        s->scratch = av_malloc(needed);
        if (!s->scratch)
            return AVERROR(ENOMEM);
        s->scratch_size = needed;

        if (s->delta_frames) {
            s->ref_frame = av_malloc(needed);
            s->delta_buf = av_malloc(needed);
            if (!s->ref_frame || !s->delta_buf)
                return AVERROR(ENOMEM);
            s->ref_frame_size = s->delta_buf_size = needed;
        }

        // avctx->thread_count is already resolved to a real, positive
        // value by this point (confirmed empirically: ff_thread_init runs
        // before this encoder's own init() callback, regardless of
        // whether AV_CODEC_CAP_FRAME_THREADS is declared -- that capability
        // only affects whether the *generic round-robin* framework wraps
        // this encoder, not whether thread_count itself gets resolved).
        s->nthreads = avctx->thread_count > 0 ? avctx->thread_count : 1;
        s->ntasks   = s->nthreads + MIKU_TASK_SLACK;

        s->tasks = av_calloc(s->ntasks, sizeof(*s->tasks));
        s->workers = av_calloc(s->nthreads, sizeof(*s->workers));
        if (!s->tasks || !s->workers)
            return AVERROR(ENOMEM);

        for (unsigned i = 0; i < s->ntasks; i++) {
            s->tasks[i].pixels = av_malloc(needed);
            if (!s->tasks[i].pixels)
                return AVERROR(ENOMEM);
        }
    }

    ret = ff_pthread_init(s, miku_enc_ctx_offsets);
    if (ret < 0)
        return ret;

    for (s->workers_started = 0; s->workers_started < s->nthreads; s->workers_started++) {
        ret = pthread_create(&s->workers[s->workers_started], NULL, miku_worker, avctx);
        if (ret) {
            // Signal exit and let already-started workers unblock and
            // return before close() joins them -- close() runs the same
            // shutdown path regardless of how many actually started.
            atomic_store(&s->exit, 1);
            av_log(avctx, AV_LOG_ERROR, "Failed to create worker thread %d/%d\n",
                   s->workers_started + 1, s->nthreads);
            return AVERROR(ret);
        }
    }

    return 0;
}

// Runs the sequential producer step (delta computation, adaptive heuristic,
// block-cap, reference update) for the current frame and fills task slot
// `s->task_index` with whichever buffer (raw or delta) the heuristic
// chose. Everything here touches only single-threaded, main-thread-only
// state -- see MikuEncContext's own comment on why none of it needs
// locking. Does not dispatch (does not touch task_fifo_mutex/task_index
// itself) -- the caller does that once this returns, matching
// frame_thread_encoder.c's own split between filling a slot and
// publishing it.
static int miku_prepare_task(AVCodecContext *avctx, const AVFrame *frame)
{
    MikuEncContext *s = avctx->priv_data;
    int stride = avctx->width * 4;
    int64_t needed = (int64_t)stride * avctx->height;
    int encode_as_delta = 0;
    MikuTask *task = &s->tasks[s->task_index];

    av_image_copy_plane(s->scratch, stride, frame->data[0], frame->linesize[0],
                         stride, avctx->height);

    if (s->delta_frames && s->have_reference) {
        const uint8_t *curr = s->scratch;
        const uint8_t *ref  = s->ref_frame;
        uint8_t *delta      = s->delta_buf;
        int64_t zero_count  = 0;

        for (int64_t i = 0; i < needed; i++) {
            delta[i] = (uint8_t)(curr[i] - ref[i]); // mod 256: plain uint8_t wraparound
            if (!delta[i]) zero_count++;
        }

        encode_as_delta = (s->frames_in_block < 100) &&
            ((double)zero_count / (double)needed >= s->delta_threshold);
    }

    if (s->delta_frames) {
        memcpy(s->ref_frame, s->scratch, needed);
        s->have_reference = 1;
    }

    if (encode_as_delta) {
        memcpy(task->pixels, s->delta_buf, needed);
        s->frames_in_block++;
    } else {
        memcpy(task->pixels, s->scratch, needed);
        s->frames_in_block = 1;
    }
    task->is_keyframe = !encode_as_delta;
    task->pts         = frame->pts;
    task->duration    = frame->duration;
    task->finished     = 0;
    task->out_data     = NULL;
    task->out_size     = 0;
    task->encode_failed = 0;

    return 0;
}

static int miku_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                              const AVFrame *frame, int *got_packet)
{
    MikuEncContext *s = avctx->priv_data;
    MikuTask *outtask;
    int ret;

    if (frame) {
        ret = miku_prepare_task(avctx, frame);
        if (ret < 0)
            return ret;

        pthread_mutex_lock(&s->task_fifo_mutex);
        s->task_index = (s->task_index + 1) % s->ntasks;
        pthread_cond_signal(&s->task_fifo_cond);
        pthread_mutex_unlock(&s->task_fifo_mutex);
    }

    // Retrieval: always the oldest not-yet-retrieved task, so packets come
    // back in exactly submission order regardless of which worker finished
    // which job when -- this, plus every worker computing byte-for-byte
    // the same thing from the same input regardless of pool size, is what
    // makes output byte-identical across thread counts. Backpressure
    // formula adopted directly from frame_thread_encoder.c's
    // ff_thread_video_encode_frame: don't block for a result unless either
    // there's nothing left to submit (frame == NULL, i.e. flushing) or the
    // queue is already as full as it's allowed to get.
    outtask = &s->tasks[s->finished_task_index];
    pthread_mutex_lock(&s->finished_task_mutex);
    if (s->task_index == s->finished_task_index ||
        (frame && !outtask->finished &&
         (s->task_index - s->finished_task_index + s->ntasks) % s->ntasks <= (unsigned)s->nthreads)) {
        pthread_mutex_unlock(&s->finished_task_mutex);
        *got_packet = 0;
        return 0;
    }
    while (!outtask->finished)
        pthread_cond_wait(&s->finished_task_cond, &s->finished_task_mutex);
    pthread_mutex_unlock(&s->finished_task_mutex);

    // We now own outtask completely -- no worker touches it again until
    // the main thread dispatches a new job into this same slot, which
    // (per the backpressure formula above) can't happen until after this
    // function returns.
    if (outtask->encode_failed) {
        av_log(avctx, AV_LOG_ERROR, "libwebp lossless encode failed (error %d)\n",
               outtask->webp_error_code);
        outtask->finished = 0;
        s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
        return AVERROR_EXTERNAL;
    }

    ret = av_new_packet(pkt, outtask->out_size);
    if (ret < 0) {
        WebPMemoryWriter tmp = { .mem = outtask->out_data, .size = outtask->out_size,
                                  .max_size = outtask->out_size };
        WebPMemoryWriterClear(&tmp);
        outtask->out_data = NULL;
        outtask->out_size = 0;
        outtask->finished = 0;
        s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
        return ret;
    }
    memcpy(pkt->data, outtask->out_data, outtask->out_size);
    {
        WebPMemoryWriter tmp = { .mem = outtask->out_data, .size = outtask->out_size,
                                  .max_size = outtask->out_size };
        WebPMemoryWriterClear(&tmp);
        outtask->out_data = NULL;
        outtask->out_size = 0;
    }
    pkt->pts = pkt->dts = outtask->pts;
    pkt->duration = outtask->duration;

    if (outtask->is_keyframe) {
        // Either delta_frames is off (every frame independent, unchanged
        // from before delta mode existed) or this frame started a fresh
        // block -- either way it's a standalone, independently decodable
        // image.
        pkt->flags |= AV_PKT_FLAG_KEY;
    } else {
        // Delta-chain frame. NOT signaled via AV_PKT_FLAG_KEY: verified
        // that AV_CODEC_ID_WEBP's codec descriptor is AV_CODEC_PROP_INTRA_ONLY,
        // which makes libavcodec/encode.c's generic pipeline force that flag
        // onto every packet this encoder produces regardless of what's set
        // here. See AV_PKT_DATA_MIKU_IS_DELTA's comment in packet.h.
        if (!av_packet_new_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, 1)) {
            av_log(avctx, AV_LOG_ERROR, "Failed to attach delta-frame side data\n");
            outtask->finished = 0;
            s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
            return AVERROR(ENOMEM);
        }
    }

    outtask->finished = 0;
    s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
    *got_packet = 1;
    return 0;
}

static av_cold int miku_encode_close(AVCodecContext *avctx)
{
    MikuEncContext *s = avctx->priv_data;

    atomic_store(&s->exit, 1);
    if (s->workers) {
        pthread_mutex_lock(&s->task_fifo_mutex);
        pthread_cond_broadcast(&s->task_fifo_cond);
        pthread_mutex_unlock(&s->task_fifo_mutex);
        for (int i = 0; i < s->workers_started; i++)
            pthread_join(s->workers[i], NULL);
    }
    ff_pthread_free(s, miku_enc_ctx_offsets);

    if (s->tasks) {
        for (unsigned i = 0; i < s->ntasks; i++) {
            av_freep(&s->tasks[i].pixels);
            if (s->tasks[i].out_data) {
                WebPMemoryWriter tmp = { .mem = s->tasks[i].out_data,
                                          .size = s->tasks[i].out_size,
                                          .max_size = s->tasks[i].out_size };
                WebPMemoryWriterClear(&tmp);
            }
        }
    }
    av_freep(&s->tasks);
    av_freep(&s->workers);
    av_freep(&s->scratch);
    av_freep(&s->ref_frame);
    av_freep(&s->delta_buf);
    return 0;
}

#define OFFSET(x) offsetof(MikuEncContext, x)
#define VE (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption miku_options[] = {
    { "method", "Lossless compression effort level (0=fastest, 9=slowest/best compression). "
      "Passed as the 'level' argument to libwebp's WebPConfigLosslessPreset, which maps it "
      "onto libwebp's internal method (0-6) through a non-linear preset table -- levels "
      "7-9 are the only way to reach the two best (slowest) internal settings, so capping "
      "this option at 6, as it used to be, silently made the best compression libwebp "
      "offers unreachable. libwebp's own default recommendation is level 6.",
      OFFSET(method), AV_OPT_TYPE_INT, { .i64 = 6 }, 0, 9, VE },
    { "exact", "Preserve exact RGB values under fully transparent alpha",
      OFFSET(exact), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VE },
    { "thread_level", "Let libwebp try its own internal threading for one frame's encode "
      "(separate from -threads, which parallelizes across frames)",
      OFFSET(thread_level), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VE },
    { "delta_frames", "Encode as adaptive delta chains against the previous raw frame "
      "instead of independent keyframes. On by default: the per-frame heuristic below "
      "already falls back to a keyframe whenever a delta wouldn't help, and delta_method "
      "bounds the worst-case cost when it's wrong, so this no longer risks forcing a bad "
      "encode the way it would have before those existed -- set to 0 to force every frame "
      "independent regardless of content.",
      OFFSET(delta_frames), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VE },
    { "delta_threshold", "Fraction of unchanged bytes (0..1) above which a candidate "
      "frame is encoded as a delta rather than ending the block early. NOT YET "
      "empirically tuned against real footage -- see the reply",
      OFFSET(delta_threshold), AV_OPT_TYPE_DOUBLE, { .dbl = 0.5 }, 0.0, 1.0, VE },
    { "delta_method", "Compression method (effort level) used for delta-residual frames "
      "only; keyframes always use -method. Lower than -method by default: bounds a "
      "delta candidate's worst-case encode time when delta_threshold's whole-frame "
      "statistic misjudges it, rather than letting one hard frame cost as much search "
      "time as a full keyframe -- see the reply",
      OFFSET(delta_method), AV_OPT_TYPE_INT, { .i64 = 2 }, 0, 6, VE },
    { NULL },
};

static const AVClass miku_encoder_class = {
    .class_name = "miku",
    .item_name  = av_default_item_name,
    .option     = miku_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_miku_encoder = {
    .p.name         = "miku",
    CODEC_LONG_NAME("Fast frame-parallel lossless WebP (VP8L) encoder for .miku"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_WEBP,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY,
    // AV_CODEC_CAP_DELAY: required now that this encoder genuinely buffers
    // output across the worker pool (a frame submitted now may not have its
    // packet retrievable until a later call, by design -- see the
    // backpressure formula in miku_encode_frame). Confirmed directly
    // against codec.h's own doc comment: without this flag, "the codec is
    // guaranteed to never be fed with NULL data" at all -- the generic
    // pipeline simply never calls encode() with frame=NULL, so this
    // encoder's flush-draining logic would never run and any still
    // in-flight tasks would be silently abandoned when close() runs.
    // Declaring it also means this encoder is now responsible for setting
    // pkt->pts *and* pkt->duration itself on every packet (previously
    // derived automatically by the framework from the input frame) -- see
    // MikuTask.duration and its use in miku_encode_frame's retrieval logic.
    //
    // AV_CODEC_CAP_FRAME_THREADS deliberately not declared, in either mode
    // -- this encoder now always manages its own worker-thread pool
    // internally (see this file's header comment), which fills the same
    // role for delta_frames=0 that the generic framework used to, and is
    // the only safe option for delta_frames=1. Declaring the generic
    // capability on top of this encoder's own internal threading would
    // mean the framework tries to round-robin frames across independent
    // *whole encoder instances* (each with its own worker pool) at the
    // same time this encoder is already parallelizing internally --
    // needless oversubscription at best, and for delta_frames=1 the exact
    // round-robin hazard already established as unsafe.
    CODEC_PIXFMTS(AV_PIX_FMT_BGRA),
    .p.priv_class   = &miku_encoder_class,
    .p.wrapper_name = "miku",
    .priv_data_size = sizeof(MikuEncContext),
    .init           = miku_encode_init,
    .close          = miku_encode_close,
    FF_CODEC_ENCODE_CB(miku_encode_frame),
};
