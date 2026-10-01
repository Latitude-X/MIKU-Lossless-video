#!/usr/bin/env bash
# miku-thread-bench.sh -- benchmarks the miku encoder across thread counts,
# recording wall time, speed, and per-core CPU usage for each, so you can
# see the real sweet spot on your own machine instead of guessing.
#
# Usage: ./miku-thread-bench.sh [max_threads] [duration_seconds] [resolution]
#   Defaults: max_threads=8, duration=3, resolution=1280x720
#
# Uses your ffmpeg from PATH by default; point at a specific build with:
#   FFMPEG=/path/to/your/ffmpeg ./miku-thread-bench.sh
set -Eeuo pipefail

FFMPEG="${FFMPEG:-ffmpeg}"
MAX_THREADS="${1:-8}"
DURATION="${2:-3}"
RESOLUTION="${3:-1280x720}"
OUTDIR="$(mktemp -d)"
RESULTS="$OUTDIR/results.csv"

command -v "$FFMPEG" >/dev/null 2>&1 || { echo "error: '$FFMPEG' not found. Set FFMPEG=/path/to/ffmpeg" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found (used for CPU sampling/math)" >&2; exit 1; }

echo "Working dir: $OUTDIR"
echo "threads,wall_s,speed_x,avg_cpu_pct,per_core_pct" > "$RESULTS"

SAMPLER_PID=""
cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for threads in $(seq 1 "$MAX_THREADS"); do
    echo ""
    echo "=== -threads:v $threads ==="
    SAMPLEFILE="$OUTDIR/samples_$threads.jsonl"

    python3 -c "
import json, time
def read_cpu():
    cores = {}
    with open('/proc/stat') as f:
        for line in f:
            if line.startswith('cpu') and len(line) > 3 and line[3].isdigit():
                p = line.split()
                n = list(map(int, p[1:8]))
                cores[p[0]] = [sum(n), n[3] + n[4]]
    return cores
with open('$SAMPLEFILE', 'w') as out:
    while True:
        out.write(json.dumps(read_cpu()) + chr(10)); out.flush()
        time.sleep(0.25)
" &
    SAMPLER_PID=$!
    sleep 0.3

    START=$(date +%s.%N)
    "$FFMPEG" -hide_banner -loglevel error -y \
        -f lavfi -i "testsrc2=size=${RESOLUTION}:rate=30:duration=${DURATION}" \
        -f lavfi -i "sine=frequency=440:duration=${DURATION}" \
        -c:v miku -threads:v "$threads" -delta_frames 1 -c:a flac \
        "$OUTDIR/out_$threads.miku"
    END=$(date +%s.%N)

    kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""

    python3 -c "
import json
wall = $END - $START
speed = $DURATION / wall if wall > 0 else 0
rows = [json.loads(l) for l in open('$SAMPLEFILE') if l.strip()]
if len(rows) < 2:
    line = f'$threads,{wall:.2f},{speed:.3f},NA,NA'
else:
    cores = sorted(rows[0].keys())
    pcts = []
    for c in cores:
        t0, i0 = rows[0][c]; t1, i1 = rows[-1][c]
        dt, di = t1 - t0, i1 - i0
        pcts.append(100*(1 - di/dt) if dt > 0 else 0.0)
    avg = sum(pcts) / len(pcts)
    percore = '|'.join(f'{p:.0f}' for p in pcts)
    line = f'$threads,{wall:.2f},{speed:.3f},{avg:.1f},{percore}'
print(line)
with open('$RESULTS', 'a') as f:
    f.write(line + chr(10))
"
done

echo ""
echo "================================================================"
echo "Done. Full results: $RESULTS"
echo "(per_core_pct is |-separated, one avg-utilization number per logical core)"
echo "Test files are in $OUTDIR/out_*.miku -- safe to delete the whole directory after."
