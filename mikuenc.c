// mikuenc.c — lossless WebP (VP8L) video encoder for .miku, with optional
// delta frames
//
// OVERVIEW
//
//   - A thin wrapper around libwebp (WebPPicture / WebPConfig / WebPEncode);
//     VP8L itself is not reimplemented here.
//
//   - Input is BGRA only (CODEC_PIXFMTS below), which is WebP's ARGB memory
//     layout; any other pixel format is converted by the implicit scaler that
//     FFmpeg puts in front of the encoder. Conversion is cheap next to the
//     VP8L search. Development benchmarks (not re-measured): about 1.7 ms per
//     1080p frame for swscale, against 9 ms to 2.9 s per frame for libwebp
//     depending on content and effort.
//
//   - With -delta_frames 0 every packet is a standalone VP8L image. With
//     -delta_frames 1 (the default) miku_encode_init switches avctx->codec_id
//     to AV_CODEC_ID_MIKU_DELTA and each frame is coded as one of
//         keyframe             a standalone image (AV_PKT_FLAG_KEY)
//         delta vs previous    the per-byte difference to the previous frame,
//                              mod 256, as a VP8L image
//         delta vs pooled key  the same against a remembered keyframe (see
//                              LONG-TERM REFERENCES); a chain root
//     Delta packets carry AV_PKT_DATA_MIKU_IS_DELTA side data
//     (miku_encode_frame), which mikumux.c turns into index entries.
//
// THREADING
//
//   - The encoder runs its own worker pool, for -delta_frames 0 and 1 alike.
//     AV_CODEC_CAP_FRAME_THREADS is not declared: frame threading runs
//     independent encoder instances on alternate frames, and a delta chain
//     needs the previous frame's pixels.
//
//   - Only the delta computation has to be sequential: a per-byte subtraction
//     plus the delta-or-keyframe decision. It runs in miku_prepare_task on the
//     one thread that calls encode(), so it needs no locking and is ordered
//     for free. The expensive part, WebPEncode, has no cross-frame dependency:
//     the producer fills a task slot with the pixels to compress (a raw frame
//     or a finished residual) and persistent workers compress slots in
//     parallel. miku_encode_frame returns packets strictly in submission
//     order, with the same finished_task_index bookkeeping as
//     libavcodec/frame_thread_encoder.c (see MIKU_TASK_SLACK).
//
//   - s->config is not modified after miku_encode_init, so workers read it
//     concurrently; a worker that needs different settings for a task edits
//     its own copy (see miku_worker).
//
//   - WebPConfig.thread_level (-thread_level, off by default) makes libwebp
//     parallelise inside one image. It is independent of the pool, which
//     parallelises across images. No benchmark of it is recorded here.
//
// FRAME COPIES AND exact=1
//
//   - Every frame is copied before it reaches libwebp, although BGRA already
//     matches WebP's ARGB layout. With exact=0 WebPEncode modifies pic.argb
//     in place (it zeroes RGB under fully transparent pixels), so pointing it
//     at the caller's AVFrame would corrupt that frame.
//
//   - A delta buffer stores the alpha residual in its alpha channel, so a
//     pixel whose alpha did not change has alpha 0 there. With exact=0 libwebp
//     would drop the RGB residual of every such pixel; a frame whose content
//     changed but whose alpha stayed constant decoded as a blank image. Delta
//     tasks are therefore always encoded with exact=1. Keyframes are too when
//     -delta_frames is on, because later deltas are computed against a
//     keyframe's exact pixels: if libwebp changed RGB under alpha=0, the
//     decoder's reference would differ from the encoder's there. Only with
//     -delta_frames 0 does -exact apply to keyframes as given (miku_worker).
//
//   - This file has no HDR or colour handling. The colour properties on the
//     AVCodecContext and the mastering-display / content-light side data
//     (which ff_encode_preinit copies into coded_side_data) reach the muxer
//     through the stream's codec parameters, and mikumux.c stores them in the
//     META colour block. The VP8L payload carries none of it.
//
// DELTA OR KEYFRAME
//
//   With -delta_auto 1 (the default) the choice is made per frame from a cost
//   model fitted to the encoder's own packet sizes; see the comment on
//   MikuEncContext's auto-tuning fields. With -delta_auto 0 a frame is coded
//   as a delta when its fraction of unchanged bytes reaches -delta_threshold.
//   Either way a chain holds at most 100 frames (frames_in_block); the frame
//   after that has to be a new chain root.
//
// LONG-TERM REFERENCES (-delta_refs)
//
//   A delta frame need not be a delta against the previous frame: it may be a
//   delta against any KEYFRAME the encoder still remembers, from a bounded
//   pool of recent keyframes kept as raw pixels in RAM (-delta_refs,
//   -delta_ref_mem). A returning scene (A..A B..B A..A) then costs a delta
//   against the first A instead of a fresh keyframe. It also replaces the
//   forced keyframe at the end of a 100-frame chain: the next frame becomes a
//   new chain root coded against the best remembered keyframe, unless the
//   normal delta-or-keyframe decision prefers a full keyframe. A frame coded
//   against a long-term reference is a chain root, like a keyframe.
//
//   Only keyframes can be referenced. A keyframe is a standalone image, so a
//   reader can fetch and decode it on demand from anywhere in the file and a
//   seek never walks a dependency graph: the worst case is one keyframe plus
//   one chain of at most 100 frames.
//
//   This encoder only emits references to earlier keyframes: it is a
//   single-pass streaming encoder and cannot know the future. The index
//   format and the demuxer accept a reference to any keyframe of the file.
//
//   Cost control: the producer step is sequential, so the search has to be
//   cheap. A frame is summarised, lazily, by MIKU_LT_SAMPLES sampled pixels.
//   Ranking a pooled keyframe is one comparison of two such sample vectors,
//   and a full-frame diff is run only against the single best candidate, and
//   only when the previous-frame delta is not already clearly good
//   (MIKU_LT_SKIP_ZERO) or the chain is at its cap. Decisions depend only on
//   frame content and the options, never on timing or thread count (see the
//   auto-tuning comment on MikuEncContext for the thread limit), so output
//   is byte-identical across runs and -threads values.

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

// The task ring holds nthreads + MIKU_TASK_SLACK slots. At most nthreads + 1
// tasks are outstanding at once (miku_encode_frame blocks for the oldest
// result beyond that), so the ring is never completely full, and a full ring
// cannot be mistaken for an empty one (task_index == finished_task_index).
// frame_thread_encoder.c sizes its ring the same way (MAX_THREADS + 2).
#define MIKU_TASK_SLACK 2

typedef struct MikuTask {
    // Input, set by the producer (the main thread) before dispatch. `pixels`
    // is allocated once in init() at the stream's frame size and never
    // reallocated. A slot belongs to one side at a time: the main thread while
    // it fills the slot and after it retrieves the packet, a single worker
    // from dispatch until `finished` is set. The backpressure in
    // miku_encode_frame ensures a slot is not refilled before its previous
    // packet has been retrieved.
    uint8_t *pixels;
    int is_keyframe;      // decided by the producer; workers only read it
    int64_t frame_no;     // submission-order index; keys this task's stats record
    int64_t nonzero;      // non-zero residual bytes (delta tasks only), for auto-tuning
    int64_t ref_frame_no; // >= 0: delta against that long-term reference keyframe
                          // (its frame number); -1: previous frame / keyframe
    int64_t pts;
    int64_t duration;     // copied from the input frame: with AV_CODEC_CAP_DELAY
                          // encode.c does not set the packet duration

    // Output, written by the worker and read by the main thread at retrieval.
    uint8_t *out_data;    // owned; released with WebPMemoryWriterClear at
                          // retrieval, or in close() if never retrieved
    size_t out_size;
    int encode_failed;
    int webp_error_code;

    int finished;         // guarded by finished_task_mutex
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
// result may influence a decision. The lag actually used is max(this, ntasks).
// It is fixed, so output does not depend on timing, and it does not depend on
// -threads either while ntasks <= this (at most this - MIKU_TASK_SLACK
// threads); see the auto-tuning comment on MikuEncContext. It must exceed the
// number of in-flight tasks so the record being consumed has been retrieved.
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
    WebPConfig config; // not modified after init(), so workers may read it
                       // concurrently
    int method;       // -method: effort level 0 (fastest) - 9 (best), the `level` of
                      // WebPConfigLosslessPreset; see the option text for the mapping
    int exact;        // -exact: WebPConfig.exact for keyframes (miku_worker says
                      // when it is overridden)
    int thread_level; // -thread_level: WebPConfig.thread_level (libwebp's own
                      // threading inside one image)

    int delta_frames; // -delta_frames; when set, init switches avctx->codec_id to
                      // AV_CODEC_ID_MIKU_DELTA
    double delta_threshold; // -delta_threshold: fraction of unchanged bytes (0..1)
                      // at or above which a candidate frame is coded as a delta.
                      // The auto-tuner's warm-up rule and, with -delta_auto 0,
                      // the only rule. The default (0.5) is an untuned starting
                      // value.
    int delta_method; // -delta_method: cap on libwebp's method (0-6) for delta-
                      // residual tasks; keyframes use the method -method selects.
                      // The delta decision rests on a whole-frame statistic, so
                      // a small, persistently complex region (noise, grain) can
                      // pass it and still cost a keyframe's worth of search
                      // time; the cap bounds that. See miku_worker.

    // ---- Sequential producer state: touched only by the thread that calls
    // encode(), so none of it needs locking ----

    // The current frame's raw pixels, copied from the AVFrame at the start of
    // every call; allocated once in init().
    uint8_t *scratch;
    int64_t scratch_size;

    // The previous frame's raw pixels ("frame N-1" while frame N is
    // prepared). A private copy rather than a pointer into a task slot: once
    // dispatched, a slot may be read and modified by a worker at any time
    // (WebPEncode edits its input in place).
    uint8_t *ref_frame;
    int64_t ref_frame_size;
    int have_reference; // 0 until ref_frame holds a real previous frame

    uint8_t *delta_buf; // residual of the current frame against ref_frame,
                        // (cur - ref) mod 256 per byte
    int64_t delta_buf_size;

    int frames_in_block; // frames in the current chain, counting its root (a
                         // keyframe, or a frame coded against a long-term
                         // reference). A chain holds at most 100 frames, which
                         // bounds how far a seek has to walk back to a root.

    // ---- Delta auto-tuning (-delta_auto, default on) ----
    //
    // Instead of a fixed -delta_threshold, the delta-or-keyframe decision uses
    // a cost model fitted to this stream's own packets. It needs no extra
    // WebPEncode calls and no extra pixel pass (the unchanged-byte count is
    // computed anyway). Per frame:
    //     predicted_delta_bytes = (EMA delta bytes / EMA delta nonzero) * nonzero
    //     delta iff predicted_delta_bytes <= EMA keyframe bytes * delta_margin
    // with both EMAs taken from real packets. miku_auto_wants_delta also holds
    // the warm-up rule (used until both kinds of packet have been fed back)
    // and the periodic delta probe.
    //
    // DETERMINISM: workers finish in timing-dependent order, so results cannot
    // be fed to the tuner as they arrive. Each result is stored in `hist`,
    // keyed by frame number, and the tuner reads frame N-lag's record only
    // when preparing frame N, with lag fixed at max(MIKU_AUTO_LAG, ntasks). At
    // most nthreads + 1 tasks are ever outstanding, so that record has always
    // been retrieved by then. Decisions therefore depend only on frame
    // content and the options, and output is byte-identical across runs and
    // across -threads values up to MIKU_AUTO_LAG - MIKU_TASK_SLACK threads.
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

        // Work on a copy: several workers can be inside WebPEncode at once, so
        // the shared s->config must stay read-only.
        local_config = s->config;

        // Delta residuals always use exact=1 (see the header comment). So do
        // keyframes once delta_frames is on: later deltas are computed against
        // the keyframe's exact pixels, and if libwebp changed RGB under alpha=0
        // here, the decoder's reference would differ from the encoder's at
        // those pixels, which shows up as corruption once alpha becomes
        // non-zero there later in the chain. With delta_frames off nothing is
        // computed against a keyframe, so -exact applies as given.
        if (task->is_keyframe == 0 || s->delta_frames)
            local_config.exact = 1;

        if (task->is_keyframe == 0) {
            // Delta residuals are capped at -delta_method effort. Packets are
            // returned in submission order, so one slow WebPEncode call holds
            // back every later packet however many workers are idle; the cap
            // bounds that stall.
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
        // Only BGRA is offered (CODEC_PIXFMTS), so this should not happen; it
        // guards the argb/argb_stride assumption made when calling WebPEncode.
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format (expected bgra)\n");
        return AVERROR(EINVAL);
    }

    // A delta-mode stream is announced to the muxer as AV_CODEC_ID_MIKU_DELTA
    // instead of WEBP. The ffmpeg CLI copies the encoder context into the
    // output stream's codec parameters (avcodec_parameters_from_context) once
    // the encoder is open, so this reaches mikumux.c as codecpar->codec_id.
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

        // Thread count. The ffmpeg CLI sets thread_count to 0 ("auto") unless
        // -threads is given, and for a codec with FF_CODEC_CAP_AUTO_THREADS
        // libavcodec leaves that 0 alone (see the capability comment on
        // ff_miku_encoder), so it is resolved here. Auto means one worker per
        // logical CPU, capped at MAX_AUTO_THREADS. It does not add the extra
        // thread that generic frame threading adds: these workers do nothing
        // but WebPEncode, which is compute- and cache-heavy, so a thread
        // beyond the CPU count buys nothing and can lose throughput to cache
        // contention. -threads remains the way to tune the count for a given
        // machine and content, in either direction.
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
    // Warm-up: until both a keyframe and a delta packet have been fed back
    // (roughly the first `lag` frames) use the fixed -delta_threshold rule.
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
// zero.
static int64_t miku_diff(uint8_t *out, const uint8_t *a, const uint8_t *b, int64_t needed)
{
    int64_t zero = 0;
    for (int64_t i = 0; i < needed; i++) {
        out[i] = (uint8_t)(a[i] - b[i]); // mod 256: plain uint8_t wraparound
        if (!out[i]) zero++;
    }
    return zero;
}

// The sequential producer step for the current frame: the residual against
// the previous frame, the search for a long-term reference, the delta-or-
// keyframe decision, the chain cap, and the update of ref_frame and the
// pool. It fills task slot s->task_index with the buffer to compress (a raw
// frame or a residual). Everything it touches is main-thread-only state (see
// MikuEncContext), so it takes no locks. It does not dispatch: the caller
// publishes the slot under task_fifo_mutex once this returns, the same split
// between filling a slot and publishing it as in frame_thread_encoder.c.
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

    // Called once for every frame: records are keyed by frame number, so a
    // skipped call would drop that frame's stats instead of deferring them.
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
        // A standalone image: any frame with -delta_frames 0, or the first
        // frame of a delta chain.
        pkt->flags |= AV_PKT_FLAG_KEY;
    } else {
        // A delta frame is not flagged AV_PKT_FLAG_KEY. The descriptor of
        // AV_CODEC_ID_WEBP, the id this encoder is registered with, is
        // AV_CODEC_PROP_INTRA_ONLY, so encode.c sets that flag on every packet
        // after this callback returns; that is decided in ff_encode_preinit,
        // before miku_encode_init changes codec_id. The delta kind is
        // therefore carried as AV_PKT_DATA_MIKU_IS_DELTA side data (see
        // packet.h): 1 byte (0) for a delta against the previous frame, or
        // 8 bytes, the little-endian frame number of the long-term reference
        // keyframe.
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
    { "method", "Lossless compression effort level, 0 (fastest) to 9 (slowest, smallest). "
      "It is the 'level' argument of libwebp's WebPConfigLosslessPreset, which selects "
      "libwebp's internal method (0-6) from a preset table; this encoder then sets "
      "libwebp's quality to 100 whatever the level. In libwebp 1.3.2 (the version this "
      "was checked against) levels 0-9 select methods 0, 1, 2, 3, 3, 4, 4, 4, 5, 6, so "
      "levels that share a method (3 and 4; 5, 6 and 7) give identical output.",
      OFFSET(method), AV_OPT_TYPE_INT, { .i64 = 6 }, 0, 9, VE },
    { "exact", "Preserve exact RGB values under fully transparent alpha, for keyframes. "
      "Delta residuals always do, and so do keyframes whenever -delta_frames is on (the "
      "default), because deltas are computed against a keyframe's exact pixels. The "
      "option therefore only changes anything with -delta_frames 0.",
      OFFSET(exact), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VE },
    { "thread_level", "Let libwebp try its own internal threading for one frame's encode "
      "(separate from -threads, which parallelizes across frames)",
      OFFSET(thread_level), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VE },
    { "delta_frames", "Encode as adaptive delta chains (against the previous frame, or a "
      "remembered keyframe, see -delta_refs) instead of independent keyframes. A frame is "
      "only coded as a delta when the per-frame decision (-delta_auto, -delta_threshold) "
      "expects that to help, and -delta_method bounds the encode time of a delta that "
      "turns out to be hard. 0 codes every frame as an independent keyframe.",
      OFFSET(delta_frames), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VE },
    { "delta_threshold", "Fraction of unchanged bytes (0..1) at or above which a frame is "
      "coded as a delta instead of a keyframe. With -delta_auto 1 (default) this is only "
      "the warm-up rule for roughly the first 24 frames, before the tuner has size "
      "feedback; with -delta_auto 0 it is the permanent rule",
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
    { "delta_method", "Upper limit on libwebp's method (0-6) for delta-residual frames; "
      "keyframes use the method that -method selects, and a delta frame uses the lower "
      "of the two. The default is below what the default -method selects, which bounds "
      "the encode time of a delta the whole-frame decision misjudged: one hard frame "
      "cannot cost as much search time as a full keyframe.",
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
    // AV_CODEC_CAP_DELAY: a packet is returned some calls after its frame was
    // submitted (the worker pool, see miku_encode_frame), and the generic code
    // only sends the NULL drain frame to an encoder that declares this. A
    // delayed encoder also has to set pts and duration itself, because
    // encode.c fills them in from the input frame only for encoders without
    // it; miku_encode_frame does, from MikuTask.
    //
    // AV_CODEC_CAP_FRAME_THREADS is not declared: it would run independent
    // encoder instances, each with its own worker pool, on alternate frames,
    // and a delta chain needs the previous frame's pixels (see the header
    // comment).
    //
    // AV_CODEC_CAP_OTHER_THREADS declares that the codec does its own
    // threading, as the libx264/libvpx/... wrappers do. `ffmpeg -encoders`
    // lists it as "threads", and FFmpeg's own consistency test
    // (libavcodec/tests/avcodec.c) requires one of the three threading
    // capabilities from any codec that sets FF_CODEC_CAP_AUTO_THREADS.
    //
    // FF_CODEC_CAP_AUTO_THREADS: without it, validate_thread_parameters()
    // (libavcodec/pthread.c) clamps thread_count to 1 for a codec that has
    // neither frame nor slice threading, so -threads would have no effect. With
    // it thread_count is left alone, including the 0 ("auto") the ffmpeg CLI
    // sets unless -threads is given; miku_encode_init resolves that 0 itself.
    .caps_internal  = FF_CODEC_CAP_AUTO_THREADS,
    CODEC_PIXFMTS(AV_PIX_FMT_BGRA),
    .p.priv_class   = &miku_encoder_class,
    .p.wrapper_name = "miku",
    .priv_data_size = sizeof(MikuEncContext),
    .init           = miku_encode_init,
    .close          = miku_encode_close,
    FF_CODEC_ENCODE_CB(miku_encode_frame),
};
