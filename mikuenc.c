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
//     setting for keyframes -- when delta_frames is off, where nothing
//     ever reconstructs against a keyframe's ref_frame cross-frame, so
//     there's nothing for exact=0's RGB-under-transparency freedom to
//     corrupt. When delta_frames is on, the same keyframe *is* referenced
//     that way (it seeds the block's delta chain), so miku_worker forces
//     exact=1 there too -- same underlying reason, just one hop removed
//     from the delta buffers themselves. See -exact's own option
//     description for the user-facing summary.
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
//
//   - LONG-TERM REFERENCES (-delta_refs). A delta frame no longer has to be
//     a delta against the previous frame: it may instead be a delta against
//     any KEYFRAME the encoder still remembers -- a bounded pool of the most
//     useful recent keyframes, kept as raw pixels in RAM. That is what makes
//     a returning scene (A..A B..B A..A) cheap: the second A reconstructs
//     against the first A's keyframe instead of costing a fresh keyframe. It
//     also replaces the forced keyframe every 100 frames: when a chain hits
//     its length cap, the next frame becomes a new chain root as a delta
//     against a remembered keyframe, and a full keyframe is only paid for
//     when that genuinely wouldn't be smaller.
//
//     Why only keyframes (never another delta) can be referenced: a keyframe
//     is a standalone image, so the demuxer/decoder can always fetch and
//     decode it on demand from anywhere in the file, and a seek never has to
//     walk an arbitrary dependency graph -- the worst case is still "one
//     chain of at most 100 frames plus one keyframe". A frame coded against
//     a long-term reference is a chain root, just like a keyframe.
//
//     Direction: references here always point backwards (to a keyframe that
//     was already encoded) because this is a single-pass streaming encoder
//     and cannot know the future. The container, demuxer and decoder do not
//     care -- a reference to a later keyframe is equally valid there.
//
//     Cost control: finding a reference must not slow the (single-threaded)
//     producer step. Each frame is summarised by a fixed set of ~2048 sample
//     pixels; ranking every pooled keyframe is then a few thousand compares,
//     and a full-frame diff is run only against the single best candidate,
//     and only when the previous-frame delta is not already clearly good (or
//     the chain is at its cap). Decisions depend only on frame content and
//     the options, never on timing or thread count, so output is still
//     byte-identical across runs and -threads values.

#include <inttypes.h>
#include <limits.h>
#include <stdatomic.h>
#include <string.h>

#include <webp/encode.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "libavutil/opt.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/internal.h"
#include "libavutil/cpu.h"
#include "libavutil/thread.h"
#include "libavutil/time.h"
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
    int64_t frame_no;   // submission-order index; keys this task's stats record
    int64_t nonzero;    // non-zero residual bytes (delta tasks only), for auto-tuning
    int64_t ref_frame_no; // >= 0: this delta is against that long-term reference
                          // keyframe (its frame number); -1: previous frame / keyframe
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

// Per-frame outcome, written at retrieval (main thread) and consumed
// MIKU_AUTO_LAG-or-more frames later by the delta auto-tuner. See
// miku_auto_apply / miku_auto_wants_delta.
typedef struct MikuStatRec {
    int valid;
    int is_key;
    int64_t size;    // compressed packet bytes
    int64_t nonzero; // residual non-zero bytes (delta frames)
} MikuStatRec;

// Minimum number of frames between a frame's submission and the moment its
// result is allowed to influence a decision. The actual lag is
// max(this, ntasks): a fixed lag, independent of timing and (for thread
// counts below this - 2) of -threads, is what keeps output deterministic --
// see the auto-tuning comment on MikuEncContext. Must exceed the max number
// of in-flight tasks so the record being consumed has always been retrieved.
#define MIKU_AUTO_LAG 24
// If no delta has been chosen for this many frames, force one probe (when
// the frame is at least plausibly similar to its predecessor) so the
// tuner keeps getting fresh delta-cost samples instead of locking out.
#define MIKU_AUTO_PROBE_INTERVAL 64
#define MIKU_AUTO_PROBE_MIN_ZERO 0.10
#define MIKU_AUTO_EMA_ALPHA 0.125

// ---- long-term reference search tuning ----
// Sample pixels per frame used to rank pooled keyframes. The standard error
// of the estimated unchanged-fraction is ~1% at this size.
#define MIKU_LT_SAMPLES 2048
// If the previous-frame delta already leaves at least this fraction of bytes
// unchanged there is nothing a reference could add; skip the search (unless
// the chain is at its cap and a new root is mandatory).
#define MIKU_LT_SKIP_ZERO 0.95
// A pooled keyframe must look at least this much better than the previous
// frame (estimated, fraction of unchanged bytes) to be worth an exact diff.
// Several standard errors of the estimate, so noise alone can't trigger it.
#define MIKU_LT_MARGIN_EST 0.04

// A keyframe remembered as raw pixels so later frames can be coded against it.
typedef struct MikuLTRef {
    uint8_t  *pixels;    // raw BGRA of a frame that was coded as a keyframe
    uint32_t *sig;       // its sample vector (MIKU_LT_SAMPLES-or-fewer pixels)
    int64_t   frame_no;  // that frame's number == its index entry in the file
    int64_t   last_used; // frame number that last chose it (or its own, if never)
} MikuLTRef;

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

    // ---- Delta auto-tuning (-delta_auto, default on) ----
    //
    // Replaces the fixed delta_threshold guess with a cost model learned
    // from the encoder's own output, at ~zero extra cost: no extra
    // WebPEncode calls, no extra pixel passes (zero_count is already
    // computed for the old heuristic). Per frame the decision is:
    //     predicted_delta_bytes = (EMA delta bytes / EMA delta nonzero) * nonzero
    //     delta iff predicted_delta_bytes <= EMA keyframe bytes * delta_margin
    // where both EMAs come from real packets of the current stream.
    //
    // DETERMINISM: results arrive from workers in timing-dependent order,
    // so feeding them to the tuner "as they arrive" would make output
    // differ run to run. Instead each result is stored in `hist` keyed by
    // frame number, and the tuner consumes frame N-lag's record only when
    // preparing frame N, with lag fixed (max(MIKU_AUTO_LAG, ntasks)). Since
    // the queue never holds more than nthreads+1 tasks, that record has
    // always been retrieved by then, so decisions depend only on frame
    // content and settings -- byte-identical output across runs and across
    // -threads values (up to MIKU_AUTO_LAG-2 threads).
    int delta_auto;
    double delta_margin; // delta must be predicted at most this fraction of a keyframe
    MikuStatRec *hist;
    unsigned hist_size;
    unsigned lag;
    int64_t frame_no;           // frames prepared so far
    int64_t last_delta_chosen;  // frame_no of the most recent delta decision
    double k_size; int have_k;  // EMA of keyframe packet bytes
    double d_size, d_nz; int have_d; // EMAs of delta packet bytes / its nonzero bytes
    int64_t n_key, n_delta, n_probe; // for the verbose summary at close()

    // ---- Long-term references (-delta_refs); see the header comment ----
    int lt_max;          // -delta_refs: most keyframes to remember (0 = feature off)
    int lt_mem_mb;       // -delta_ref_mem: memory budget for the remembered pixels
    int lt_cap;          // effective pool capacity after the budget; 0 = disabled
    MikuLTRef *lt;       // pool, lt_cap slots, filled in order then LRU-replaced
    int lt_n;            // slots in use
    int lt_nsamp;        // sample pixels per frame
    uint32_t *lt_pos;    // the fixed sample positions (pixel indices)
    uint32_t *cur_sig;   // current frame's sample vector
    uint8_t *delta_buf2; // residual of the current frame against the best pooled keyframe
    int64_t n_lt, n_lt_root; // frames coded against a pooled keyframe / of those, chain re-roots
    int64_t t_prod_us, t_lt_us; // diagnostics (verbose log): time in the producer step / in its long-term part

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

        // Keyframes need exact=1 too, but only once delta_frames is on:
        // this keyframe becomes the ref_frame a delta chain reconstructs
        // against, and if exact=0 lets WebP pick different RGB under
        // alpha=0 here, the encoder's true ref_frame and the decoder's
        // reconstructed one silently diverge at every pixel that was
        // transparent on this keyframe. Invisible right up until (if
        // ever) alpha moves off zero there later in the chain, at which
        // point it surfaces as real, visible corruption -- see the reply.
        // Harmless, and correctly left alone, when delta_frames is off:
        // nothing ever references this keyframe's ref_frame cross-frame
        // in that mode, so -exact keeps meaning exactly what its own
        // description says.
        if (task->is_keyframe == 0 || s->delta_frames)
            local_config.exact = 1;

        if (task->is_keyframe == 0) {
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

        if (s->delta_frames && s->lt_max > 0) {
            int64_t total_px = (int64_t)avctx->width * avctx->height;
            int64_t per      = needed + (int64_t)MIKU_LT_SAMPLES * 4;
            int64_t fit      = (((int64_t)s->lt_mem_mb) << 20) / per;

            s->lt_cap = (int)FFMIN((int64_t)s->lt_max, FFMIN(fit, (int64_t)INT_MAX));
            if (s->lt_cap <= 0) {
                s->lt_cap = 0;
                av_log(avctx, AV_LOG_WARNING,
                       "-delta_ref_mem %d MB can't hold even one %dx%d frame; "
                       "long-term references disabled\n",
                       s->lt_mem_mb, avctx->width, avctx->height);
            } else {
                uint32_t state = 0x9E3779B9u;

                s->lt_nsamp   = (int)FFMIN((int64_t)MIKU_LT_SAMPLES, total_px);
                s->lt         = av_calloc(s->lt_cap, sizeof(*s->lt));
                s->lt_pos     = av_malloc_array(s->lt_nsamp, sizeof(*s->lt_pos));
                s->cur_sig    = av_malloc_array(s->lt_nsamp, sizeof(*s->cur_sig));
                s->delta_buf2 = av_malloc(needed);
                if (!s->lt || !s->lt_pos || !s->cur_sig || !s->delta_buf2)
                    return AVERROR(ENOMEM);
                // One sample per equal-sized stratum of the frame, jittered by
                // a fixed LCG: even coverage, and no alignment with regular
                // patterns (dither, scanlines) that a plain grid could alias.
                for (int i = 0; i < s->lt_nsamp; i++) {
                    uint64_t lo = (uint64_t)total_px * i / s->lt_nsamp;
                    uint64_t hi = (uint64_t)total_px * (i + 1) / s->lt_nsamp;
                    state = state * 1664525u + 1013904223u;
                    s->lt_pos[i] = (uint32_t)(lo + state % (hi - lo));
                }
            }
        }

        // avctx->thread_count is 0 here on a completely ordinary run: the
        // ffmpeg CLI explicitly sets "auto" (fftools/ffmpeg_enc.c) unless
        // the user passes -threads themselves, and libavcodec's own
        // AVOption default for thread_count is 1 (options_table.h), not 0
        // -- neither of those is a real core count. Resolving that 0 used
        // to be this encoder's actual bug, not just a missing nicety:
        // ff_thread_init() (libavcodec/pthread.c, validate_thread_parameters)
        // force-clamps thread_count to exactly 1 for any codec that
        // declares none of AV_CODEC_CAP_FRAME_THREADS, _SLICE_THREADS, or
        // the internal FF_CODEC_CAP_AUTO_THREADS -- confirmed directly
        // against that function's current source. We deliberately don't
        // want the first two (see the capabilities comment below), so
        // without the third this encoder silently got exactly one worker
        // thread on every run regardless of how many cores were actually
        // available -- and, because the clamp fires unconditionally in
        // that branch, regardless of an explicit -threads N too.
        // FF_CODEC_CAP_AUTO_THREADS (below) stops that clamp, but it also
        // hands *us* the job that ff_frame_thread_init would otherwise
        // have done for a generically-threaded codec: turning thread_count
        // == 0 into a real number.
        //
        // Deliberately NOT mirroring ff_frame_thread_init's own "+1" here,
        // on reflection -- that extra thread earns its keep for a decode
        // pipeline with real I/O- and sync-adjacent stalls to fill; there's
        // no equivalent idle time to fill in a pool that does nothing but
        // call WebPEncode back to back. VP8L's compression search is
        // compute- *and* cache-heavy (large backward-reference hash
        // tables, real memory-bandwidth pressure), so on a CPU whose
        // logical thread count outruns its physical core count and shared
        // cache size -- a 4C/8T part, say -- one worker beyond nb_cpus
        // buys nothing and can cost real throughput once the concurrent
        // working set stops fitting the cache: more reported CPU usage
        // from cores churning on contention, less actual progress. Plain
        // nb_cpus, still capped at MAX_AUTO_THREADS, is the more
        // defensible default; -threads remains how to tune it further for
        // a specific machine and content, in either direction -- fewer
        // threads can genuinely outrun more once the cache is the limit,
        // not the core count.
        if (avctx->thread_count > 0) {
            s->nthreads = avctx->thread_count;
        } else {
            int nb_cpus = av_cpu_count();
            s->nthreads = nb_cpus > 0 ? nb_cpus : 1;
            if (s->nthreads > MAX_AUTO_THREADS)
                s->nthreads = MAX_AUTO_THREADS;
        }
        s->ntasks   = s->nthreads + MIKU_TASK_SLACK;

        if (s->delta_frames && s->delta_auto) {
            // lag >= ntasks > max in-flight tasks, so frame N-lag is always
            // retrieved before frame N is prepared. hist_size = lag + 1
            // keeps a record from being overwritten before it's consumed.
            s->lag       = s->ntasks > MIKU_AUTO_LAG ? s->ntasks : MIKU_AUTO_LAG;
            s->hist_size = s->lag + 1;
            s->hist      = av_calloc(s->hist_size, sizeof(*s->hist));
            if (!s->hist)
                return AVERROR(ENOMEM);
        }

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

static void miku_ema(double *x, int *have, double v)
{
    if (!*have) { *x = v; *have = 1; }
    else        *x += (v - *x) * MIKU_AUTO_EMA_ALPHA;
}

// Consumes the stats record of frame (frame_no - lag), if there is one.
// Called once per frame from miku_prepare_task, before the decision.
static void miku_auto_apply(MikuEncContext *s)
{
    MikuStatRec *r;
    if (s->frame_no < (int64_t)s->lag)
        return;
    r = &s->hist[(s->frame_no - s->lag) % s->hist_size];
    if (!r->valid)
        return;
    if (r->is_key) {
        miku_ema(&s->k_size, &s->have_k, (double)r->size);
    } else {
        int one = s->have_d;
        miku_ema(&s->d_size, &s->have_d, (double)r->size);
        miku_ema(&s->d_nz,   &one,       (double)r->nonzero);
    }
    r->valid = 0;
}

// The decision itself. zero_frac is the fraction of unchanged bytes vs the
// previous frame (already computed by the caller); nonzero is its
// complement in bytes. Returns 1 to encode as a delta.
static int miku_auto_wants_delta(MikuEncContext *s, double zero_frac, int64_t nonzero)
{
    // Warm-up: until both a keyframe and a delta have actually been
    // encoded and fed back (i.e. for roughly the first `lag` frames),
    // fall back to the old fixed-threshold rule.
    if (!s->have_k || !s->have_d)
        return zero_frac >= s->delta_threshold;

    {
        double bytes_per_nz = s->d_size / (s->d_nz > 1.0 ? s->d_nz : 1.0);
        double predicted    = bytes_per_nz * (double)nonzero;
        if (predicted <= s->k_size * s->delta_margin)
            return 1;
    }

    // Keep the delta-cost estimate fresh: if the model hasn't picked a
    // delta in a while, probe one on a plausibly-similar frame. Cost is one
    // cheap (delta_method) encode per MIKU_AUTO_PROBE_INTERVAL frames.
    if (s->frame_no - s->last_delta_chosen >= MIKU_AUTO_PROBE_INTERVAL &&
        zero_frac >= MIKU_AUTO_PROBE_MIN_ZERO) {
        s->n_probe++;
        return 1;
    }
    return 0;
}

// ---- long-term reference pool -------------------------------------------

// Fills the current frame's sample vector: the fixed sample pixels, packed.
static void miku_lt_sample(const MikuEncContext *s, const uint8_t *px, uint32_t *sig)
{
    for (int i = 0; i < s->lt_nsamp; i++)
        sig[i] = AV_RN32A(px + 4 * (int64_t)s->lt_pos[i]);
}

// Number of byte positions that are equal between two sample vectors -- the
// sampled counterpart of "zero residual bytes", i.e. of the exact statistic
// the full-frame diff pass computes.
static int64_t miku_lt_equal_bytes(const uint32_t *a, const uint32_t *b, int n)
{
    int64_t z = 0;
    for (int i = 0; i < n; i++) {
        uint32_t x = a[i] ^ b[i];
        // high bit of each byte of y is set iff that byte of x is zero
        uint32_t y = ~(((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x | 0x7F7F7F7Fu);
        z += av_popcount(y);
    }
    return z;
}

// Ranks every pooled keyframe against the current frame by sampled equality
// and returns the best slot (ties go to the most recent keyframe), or -1 if
// the pool is empty. *est is its estimated fraction of unchanged bytes.
static int miku_lt_best(const MikuEncContext *s, double *est)
{
    int best = -1;
    int64_t best_z = -1;

    for (int i = 0; i < s->lt_n; i++) {
        int64_t z = miku_lt_equal_bytes(s->cur_sig, s->lt[i].sig, s->lt_nsamp);
        if (z > best_z || (z == best_z && s->lt[i].frame_no > s->lt[best].frame_no)) {
            best   = i;
            best_z = z;
        }
    }
    if (best >= 0)
        *est = (double)best_z / (4.0 * (double)s->lt_nsamp);
    return best;
}

// Remembers `pixels` (a frame that is being coded as a keyframe) as a
// long-term reference. When the pool is full the least recently used slot is
// replaced; ties go to the older keyframe. Slot buffers are allocated on first
// use and then reused, so a stream that never fills the pool never pays for
// the whole budget. Failure just means "not remembered" -- never an error.
static void miku_lt_insert(MikuEncContext *s, const uint8_t *pixels, int64_t needed,
                           int64_t frame_no)
{
    MikuLTRef *r;

    if (s->lt_n < s->lt_cap) {
        r = &s->lt[s->lt_n];
        if (!r->pixels) {
            r->pixels = av_malloc(needed);
            r->sig    = av_malloc_array(s->lt_nsamp, sizeof(*r->sig));
            if (!r->pixels || !r->sig) {
                av_freep(&r->pixels);
                av_freep(&r->sig);
                return;
            }
        }
        s->lt_n++;
    } else {
        int victim = 0;
        for (int i = 1; i < s->lt_n; i++) {
            if (s->lt[i].last_used < s->lt[victim].last_used ||
                (s->lt[i].last_used == s->lt[victim].last_used &&
                 s->lt[i].frame_no < s->lt[victim].frame_no))
                victim = i;
        }
        r = &s->lt[victim];
    }
    memcpy(r->pixels, pixels, needed);
    memcpy(r->sig, s->cur_sig, (size_t)s->lt_nsamp * sizeof(*r->sig));
    r->frame_no  = frame_no;
    r->last_used = frame_no;
}

// out = a - b (mod 256) over `needed` bytes; returns how many result bytes are
// zero. The same loop the previous-frame diff has always used.
static int64_t miku_diff(uint8_t *out, const uint8_t *a, const uint8_t *b, int64_t needed)
{
    int64_t zero = 0;
    for (int64_t i = 0; i < needed; i++) {
        out[i] = (uint8_t)(a[i] - b[i]); // mod 256: plain uint8_t wraparound
        if (!out[i]) zero++;
    }
    return zero;
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
//
// Each frame ends up as exactly one of:
//   keyframe            standalone image (also seeds the long-term pool)
//   delta vs previous   the classic chain link
//   delta vs pooled key a chain ROOT: needs only that keyframe, nothing before
static int miku_prepare_task(AVCodecContext *avctx, const AVFrame *frame)
{
    MikuEncContext *s = avctx->priv_data;
    int stride = avctx->width * 4;
    int64_t needed = (int64_t)stride * avctx->height;
    int encode_as_delta = 0;
    int lt_choice = -1;      // pool slot when coded against a long-term reference
    int chain_ok = 1;        // may this frame be a plain delta vs the previous one?
    MikuTask *task = &s->tasks[s->task_index];
    int64_t t_start = av_gettime_relative(), t_mark;
    int sig_ready;           // has cur_sig been filled for this frame yet?

    av_image_copy_plane(s->scratch, stride, frame->data[0], frame->linesize[0],
                         stride, avctx->height);

    // Must run exactly once per frame, unconditionally: records are keyed
    // by frame number, so a skipped call would silently drop that frame's
    // stats rather than deferring them.
    if (s->delta_frames && s->delta_auto)
        miku_auto_apply(s);
    task->nonzero      = 0;
    task->ref_frame_no = -1;

    // The sample vector is only needed to search the pool or to remember this
    // frame in it, and the sampling itself (thousands of scattered reads, one
    // cache line and TLB entry each) is not free -- so it is taken on first
    // use. In the common case of a long run of small deltas neither happens,
    // and long-term references cost this frame nothing.
    sig_ready = 0;

    if (s->delta_frames && s->have_reference) {
        int64_t zero_count = miku_diff(s->delta_buf, s->scratch, s->ref_frame, needed);
        int64_t cand_nz    = needed - zero_count;   // changed bytes of the chosen delta
        double  zp         = (double)zero_count / (double)needed;

        chain_ok = s->frames_in_block < 100;

        // Long-term candidate: only worth looking for when the previous-frame
        // delta isn't already clearly good, or when the chain is at its cap and
        // a new root is mandatory (then it is the alternative to a keyframe).
        if (s->lt_n > 0 && (!chain_ok || zp < MIKU_LT_SKIP_ZERO)) {
            double est = 0.0;
            int b;
            t_mark = av_gettime_relative();
            miku_lt_sample(s, s->scratch, s->cur_sig);
            sig_ready = 1;
            b = miku_lt_best(s, &est);
            if (b >= 0 && (!chain_ok || est > zp + MIKU_LT_MARGIN_EST)) {
                int64_t zl   = miku_diff(s->delta_buf2, s->scratch, s->lt[b].pixels, needed);
                int64_t nz_l = needed - zl;
                // Take it if the chain forces a root, or if it changes at
                // least 10% fewer bytes than the previous-frame delta.
                if (!chain_ok || nz_l * 10 < cand_nz * 9) {
                    lt_choice = b;
                    cand_nz   = nz_l;
                }
            }
            s->t_lt_us += av_gettime_relative() - t_mark;
        }

        if (chain_ok || lt_choice >= 0) {
            double zf = 1.0 - (double)cand_nz / (double)needed;
            if (s->delta_auto)
                encode_as_delta = miku_auto_wants_delta(s, zf, cand_nz);
            else
                encode_as_delta = zf >= s->delta_threshold;
        }
        if (!encode_as_delta)
            lt_choice = -1;     // a keyframe beat both kinds of delta
        task->nonzero = cand_nz;
    }

    if (s->delta_frames) {
        memcpy(s->ref_frame, s->scratch, needed);
        s->have_reference = 1;
    }

    if (lt_choice >= 0) {
        memcpy(task->pixels, s->delta_buf2, needed);
        s->frames_in_block = 1; // a chain root, like a keyframe
        task->ref_frame_no = s->lt[lt_choice].frame_no;
        s->lt[lt_choice].last_used = s->frame_no;
        s->n_lt++;
        if (!chain_ok)
            s->n_lt_root++;
    } else if (encode_as_delta) {
        memcpy(task->pixels, s->delta_buf, needed);
        s->frames_in_block++;
    } else {
        memcpy(task->pixels, s->scratch, needed);
        s->frames_in_block = 1;
        if (s->lt_cap) {
            t_mark = av_gettime_relative();
            if (!sig_ready)
                miku_lt_sample(s, s->scratch, s->cur_sig);
            miku_lt_insert(s, s->scratch, needed, s->frame_no);
            s->t_lt_us += av_gettime_relative() - t_mark;
        }
    }
    task->is_keyframe = !encode_as_delta;
    task->frame_no    = s->frame_no;
    if (encode_as_delta) {
        s->last_delta_chosen = s->frame_no;
        s->n_delta++;
    } else {
        s->n_key++;
    }
    s->frame_no++;
    task->pts         = frame->pts;
    task->duration    = frame->duration;
    task->finished     = 0;
    task->out_data     = NULL;
    task->out_size     = 0;
    task->encode_failed = 0;

    s->t_prod_us += av_gettime_relative() - t_start;
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

    if (s->hist) {
        // Record this frame's real outcome for the auto-tuner; consumed
        // `lag` frames later (see MikuEncContext) so timing never leaks in.
        MikuStatRec *r = &s->hist[outtask->frame_no % s->hist_size];
        r->is_key  = outtask->is_keyframe;
        r->size    = (int64_t)outtask->out_size;
        r->nonzero = outtask->nonzero;
        r->valid   = 1;
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
        // Payload: 1 byte (0) = delta against the previous frame; 8 bytes =
        // little-endian frame number of the long-term reference keyframe.
        size_t sd_size = outtask->ref_frame_no >= 0 ? 8 : 1;
        uint8_t *sd = av_packet_new_side_data(pkt, AV_PKT_DATA_MIKU_IS_DELTA, sd_size);
        if (!sd) {
            av_log(avctx, AV_LOG_ERROR, "Failed to attach delta-frame side data\n");
            outtask->finished = 0;
            s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
            return AVERROR(ENOMEM);
        }
        if (sd_size == 8)
            AV_WL64(sd, (uint64_t)outtask->ref_frame_no);
        else
            sd[0] = 0;
    }

    outtask->finished = 0;
    s->finished_task_index = (s->finished_task_index + 1) % s->ntasks;
    *got_packet = 1;
    return 0;
}

static av_cold int miku_encode_close(AVCodecContext *avctx)
{
    MikuEncContext *s = avctx->priv_data;

    if (s->delta_frames && s->lt_cap && s->frame_no > 0)
        av_log(avctx, AV_LOG_VERBOSE,
               "long-term refs: %"PRId64" frames coded against a remembered keyframe "
               "(%"PRId64" of them re-rooting a full chain instead of a keyframe); "
               "pool %d/%d keyframes\n", s->n_lt, s->n_lt_root, s->lt_n, s->lt_cap);

    if (s->delta_frames && s->frame_no > 0)
        av_log(avctx, AV_LOG_VERBOSE,
               "producer step (the one sequential part): %.2f ms/frame, of which long-term "
               "reference work %.3f ms/frame\n",
               (double)s->t_prod_us / 1000.0 / (double)s->frame_no,
               (double)s->t_lt_us   / 1000.0 / (double)s->frame_no);

    if (s->delta_frames && s->delta_auto && s->frame_no > 0)
        av_log(avctx, AV_LOG_VERBOSE,
               "delta auto-tune: %"PRId64" keyframes, %"PRId64" delta frames "
               "(%"PRId64" probes); est. keyframe %.0f B, delta %.3f B per "
               "non-zero byte\n", s->n_key, s->n_delta, s->n_probe, s->k_size,
               s->d_nz > 1.0 ? s->d_size / s->d_nz : 0.0);

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
    av_freep(&s->hist);
    if (s->lt) {
        for (int i = 0; i < s->lt_cap; i++) {
            av_freep(&s->lt[i].pixels);
            av_freep(&s->lt[i].sig);
        }
    }
    av_freep(&s->lt);
    av_freep(&s->lt_pos);
    av_freep(&s->cur_sig);
    av_freep(&s->delta_buf2);
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
    { "exact", "Preserve exact RGB values under fully transparent alpha, for keyframes. "
      "Always on regardless of this setting for delta residuals, and for keyframes too "
      "once -delta_frames is on (the default) -- see miku_worker's own comment for why a "
      "keyframe that seeds a delta chain needs this the same way the deltas themselves "
      "do. Only actually controls anything when -delta_frames 0 is set explicitly.",
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
      "frame is encoded as a delta rather than ending the block early. With "
      "-delta_auto 1 (default) this is only the warm-up rule used for the first "
      "~24 frames, before the tuner has real size feedback; with -delta_auto 0 it "
      "is the permanent rule",
      OFFSET(delta_threshold), AV_OPT_TYPE_DOUBLE, { .dbl = 0.5 }, 0.0, 1.0, VE },
    { "delta_auto", "Choose delta vs keyframe per frame from a cost model learned "
      "from this stream's own compressed packet sizes (no extra encodes, "
      "deterministic across runs and -threads), instead of the fixed "
      "-delta_threshold. Only meaningful with -delta_frames 1",
      OFFSET(delta_auto), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VE },
    { "delta_margin", "With -delta_auto: encode as delta only if its predicted size "
      "is at most this fraction of the estimated keyframe size. Below 1 biases "
      "toward keyframes, trading some file size for cheaper seeking/decoding "
      "(no dependency chain)",
      OFFSET(delta_margin), AV_OPT_TYPE_DOUBLE, { .dbl = 0.85 }, 0.1, 2.0, VE },
    { "delta_refs", "Long-term references: let a delta frame reconstruct against any of "
      "up to this many remembered earlier KEYFRAMES, not only the previous frame. Makes "
      "a returning scene (A B A) cheap, and replaces the forced keyframe every 100 frames "
      "with a delta against a remembered keyframe whenever that is smaller. Each remembered "
      "keyframe costs one raw frame of RAM (see -delta_ref_mem). 0 turns the feature off; "
      "output is then exactly the older format, readable by older demuxers. Files that "
      "actually use a long-term reference are written as format v3 (the muxer decides)",
      OFFSET(lt_max), AV_OPT_TYPE_INT, { .i64 = 32 }, 0, 4096, VE },
    { "delta_ref_mem", "Memory budget in MB for the raw keyframes remembered as long-term "
      "references; the pool holds at most min(-delta_refs, this budget / frame size) "
      "keyframes. Memory is only committed as keyframes are actually remembered",
      OFFSET(lt_mem_mb), AV_OPT_TYPE_INT, { .i64 = 256 }, 1, 16384, VE },
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
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY |
                      AV_CODEC_CAP_OTHER_THREADS,
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
    //
    // AV_CODEC_CAP_OTHER_THREADS above, by contrast, is exactly what we
    // do want: the public "this codec threads itself, through neither
    // generic mechanism" declaration -- the same one every external-
    // library-wrapping encoder in this tree pairs with
    // FF_CODEC_CAP_AUTO_THREADS below (libx264.c, libx265.c, libvpxenc.c,
    // libaomenc.c, librav1e.c, libsvtav1.c, ...; verified against a real
    // checkout, not assumed). On its own it doesn't change
    // ff_thread_init()'s behavior, though -- see .caps_internal.
    .caps_internal  = FF_CODEC_CAP_AUTO_THREADS,
    // The flag that actually matters for thread_count. validate_thread_
    // parameters() (libavcodec/pthread.c) force-clamps thread_count to 1
    // for any codec lacking AV_CODEC_CAP_FRAME_THREADS, _SLICE_THREADS,
    // *and* this internal flag -- confirmed directly against that
    // function's source. Without it, this encoder always got exactly one
    // worker thread, on every run, regardless of -threads or core count:
    // the actual cause of a real, reported "encoder only ever uses one
    // core" bug. Declaring it tells the generic pipeline this codec -- not
    // it -- is responsible for interpreting thread_count == 0 ("auto"),
    // which is why miku_encode_init now resolves that 0 itself (see its
    // own comment) rather than leaving it to a generic frame/slice-thread
    // init that never runs for a codec declaring neither.
    CODEC_PIXFMTS(AV_PIX_FMT_BGRA),
    .p.priv_class   = &miku_encoder_class,
    .p.wrapper_name = "miku",
    .priv_data_size = sizeof(MikuEncContext),
    .init           = miku_encode_init,
    .close          = miku_encode_close,
    FF_CODEC_ENCODE_CB(miku_encode_frame),
};
