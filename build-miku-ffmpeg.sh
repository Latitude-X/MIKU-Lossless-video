#!/usr/bin/env bash
#miku-help-start
# build-miku-ffmpeg.sh -- build and install an FFmpeg with .miku container
# and codec support (mikuenc.c, mikudeltadec.c, mikudec.c, mikumux.c,
# miku-support.patch) on top of a fresh upstream FFmpeg checkout.
#
# Run this from the directory containing those five files, e.g.:
#   cd /path/to/miku-source && ./build-miku-ffmpeg.sh
#
# It's safe to re-run: by default it reuses and updates its build directory
# instead of starting from scratch, so later runs are just "pull the latest
# FFmpeg, re-apply the patch, rebuild, reinstall."
#
# Options:
#   --ref <git-ref>          FFmpeg git ref to build: a branch, tag, or full
#                             commit hash. Default: master.
#                             If the patch fails to apply (upstream FFmpeg
#                             drifts constantly), try pinning an older
#                             release tag here, e.g. --ref n7.1.
#   --repo <url>              FFmpeg git remote to clone from.
#                             Default: https://github.com/FFmpeg/FFmpeg.git
#   --build-dir <path>        Where to check out and build FFmpeg.
#                             Default: <source-dir>/ffmpeg-miku-build
#   --source-dir <path>       Directory containing the miku*.c/.patch files.
#                             Default: this script's own directory.
#   --jobs <N>                 Parallel make jobs. Default: nproc.
#   --prefix <path>            Passed through as ./configure --prefix.
#                             Default: FFmpeg's own default (/usr/local).
#   --extra-configure-args "<args>"
#                             Extra flags appended to ./configure, split on
#                             whitespace. Quote paths without spaces in them.
#   --clean                    Wipe the build directory first instead of
#                             reusing/updating it.
#   --no-install                Build only; skip "sudo make install".
#   --no-deps                   Skip the OS package-manager dependency step
#                             (use if you've already got everything).
#   -h, --help                     Show this help and exit.
#
# Environment variables (same effect as the matching flag; flags win):
#   FFMPEG_REF, FFMPEG_REPO, BUILD_DIR, SOURCE_DIR, JOBS, PREFIX,
#   EXTRA_CONFIGURE_ARGS
#
# Supported package managers (auto-detected): apt, dnf, yum, pacman,
# zypper, apk, xbps. Anything else: it'll warn and expect you to have
# build tools, git, curl/wget, pkg-config and libwebp dev headers already.
#
# Fallbacks built in: git clone -> shallow branch/tag clone -> full clone +
# checkout -> source tarball download, in that order, for fetching FFmpeg.
# For libwebp specifically (a hard requirement for the miku codec): OS
# package first, then building it from its own upstream source as a last
# resort. For applying miku-support.patch: git apply -> patch -p1 ->
# patch -p1 with fuzz, in that order.
#miku-help-end
set -Eeuo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "error: this script needs bash (run it as: bash $0)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Defaults (env var, then flag -- flags win). BUILD_DIR and JOBS are
# resolved after flag parsing since they can depend on SOURCE_DIR/nproc.
# ---------------------------------------------------------------------------
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

SOURCE_DIR="${SOURCE_DIR:-}"
SOURCE_DIR_EXPLICIT=0
[ -n "$SOURCE_DIR" ] && SOURCE_DIR_EXPLICIT=1
FFMPEG_REPO="${FFMPEG_REPO:-https://github.com/FFmpeg/FFmpeg.git}"
FFMPEG_REF="${FFMPEG_REF:-master}"
BUILD_DIR="${BUILD_DIR:-}"
JOBS="${JOBS:-}"
PREFIX="${PREFIX:-}"
EXTRA_CONFIGURE_ARGS="${EXTRA_CONFIGURE_ARGS:-}"
DO_INSTALL=1
DO_DEPS=1
DO_CLEAN=0
SUDO=""

REQUIRED_FILES=(mikuenc.c mikudeltadec.c mikudec.c mikumux.c miku-support.patch)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[34m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi

log()  { printf '%s[miku]%s %s\n' "$C_BLUE$C_BOLD" "$C_RESET" "$*"; }
ok()   { printf '%s[miku]%s %s%s%s\n' "$C_GREEN$C_BOLD" "$C_RESET" "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s[miku] warning:%s %s\n' "$C_YELLOW$C_BOLD" "$C_RESET" "$*" >&2; }
err()  { printf '%s[miku] error:%s %s\n' "$C_RED$C_BOLD" "$C_RESET" "$*" >&2; }
die()  { err "$*"; exit 1; }

on_error() {
    local exit_code=$? line=$1
    err "failed at line $line (exit code $exit_code)."
    err "nothing after this point ran. It's safe to fix the issue and re-run --"
    err "this script reuses \"$BUILD_DIR\" instead of starting over from scratch."
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

print_help() {
    sed -n '/^#miku-help-start$/,/^#miku-help-end$/p' "$SCRIPT_PATH" \
        | sed -e '1d;$d' -e 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --ref)                   FFMPEG_REF="$2"; shift 2 ;;
        --repo)                  FFMPEG_REPO="$2"; shift 2 ;;
        --build-dir)              BUILD_DIR="$2"; shift 2 ;;
        --source-dir)             SOURCE_DIR="$2"; SOURCE_DIR_EXPLICIT=1; shift 2 ;;
        --jobs)                   JOBS="$2"; shift 2 ;;
        --prefix)                 PREFIX="$2"; shift 2 ;;
        --extra-configure-args)   EXTRA_CONFIGURE_ARGS="$2"; shift 2 ;;
        --clean)                  DO_CLEAN=1; shift ;;
        --no-install)             DO_INSTALL=0; shift ;;
        --no-deps)                DO_DEPS=0; shift ;;
        -h|--help)                print_help; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

dir_has_required_files() {
    local d="$1" f
    for f in "${REQUIRED_FILES[@]}"; do
        [ -f "$d/$f" ] || return 1
    done
    return 0
}

if [ "$SOURCE_DIR_EXPLICIT" -eq 0 ]; then
    if dir_has_required_files "$SCRIPT_DIR"; then
        SOURCE_DIR="$SCRIPT_DIR"
    elif dir_has_required_files "$PWD"; then
        SOURCE_DIR="$PWD"
        log "found the miku source files in the current directory ($PWD); this script itself lives in $SCRIPT_DIR"
    else
        SOURCE_DIR="$SCRIPT_DIR"   # neither has everything; preflight() reports both candidates clearly
    fi
fi

[ -n "$BUILD_DIR" ] || BUILD_DIR="$SOURCE_DIR/ffmpeg-miku-build"
[ -n "$JOBS" ] || JOBS="$(nproc 2>/dev/null || echo 2)"
case "$JOBS" in ''|*[!0-9]*) die "--jobs must be a positive integer, got: $JOBS" ;; esac

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight() {
    log "miku FFmpeg build script starting"
    log "  source dir : $SOURCE_DIR"
    log "  build dir  : $BUILD_DIR"
    log "  ffmpeg ref : $FFMPEG_REF"
    log "  jobs       : $JOBS"

    if [ "$(uname -s)" != "Linux" ]; then
        die "this script targets Linux; detected $(uname -s), which it won't know how to install packages on."
    fi

    local missing=() f
    for f in "${REQUIRED_FILES[@]}"; do
        [ -f "$SOURCE_DIR/$f" ] || missing+=("$f")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        if [ "$SOURCE_DIR_EXPLICIT" -eq 1 ]; then
            die "missing source file(s) in $SOURCE_DIR: ${missing[*]}
That's the --source-dir (or SOURCE_DIR env var) you gave it -- double check the path and file names."
        else
            die "missing source file(s) in $SOURCE_DIR: ${missing[*]}
Checked both this script's own directory ($SCRIPT_DIR) and the current directory ($PWD); neither had a complete set.
Put the miku*.c/.patch files next to this script (or cd into whichever folder actually has them and run it from there), or pass --source-dir <path> explicitly."
        fi
    fi

    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
        warn "running as root already; will not use sudo for install steps."
    elif command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        SUDO=""
        warn "not root and no sudo found; package install and 'make install' will likely fail unless you run this whole script as root."
    fi
}

# ---------------------------------------------------------------------------
# OS dependencies
# ---------------------------------------------------------------------------
install_deps() {
    if [ "$DO_DEPS" -eq 0 ]; then
        log "skipping OS dependency install (--no-deps)"
        return 0
    fi

    log "detecting package manager..."
    local pm=""
    if   command -v apt-get      >/dev/null 2>&1; then pm=apt
    elif command -v dnf          >/dev/null 2>&1; then pm=dnf
    elif command -v yum          >/dev/null 2>&1; then pm=yum
    elif command -v pacman       >/dev/null 2>&1; then pm=pacman
    elif command -v zypper       >/dev/null 2>&1; then pm=zypper
    elif command -v apk          >/dev/null 2>&1; then pm=apk
    elif command -v xbps-install >/dev/null 2>&1; then pm=xbps
    fi

    if [ -z "$pm" ]; then
        warn "couldn't identify a known package manager (apt/dnf/yum/pacman/zypper/apk/xbps)."
        warn "make sure you already have: a C compiler + make, git, curl or wget, pkg-config, and libwebp development headers."
        return 0
    fi
    log "package manager: $pm"

    local pkgs=() p
    case "$pm" in
        apt)
            pkgs=(build-essential git curl wget pkg-config yasm nasm libwebp-dev ca-certificates)
            $SUDO apt-get update -y || warn "apt-get update failed; continuing with whatever's cached"
            if ! $SUDO apt-get install -y "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package so one bad name doesn't block the rest"
                for p in "${pkgs[@]}"; do $SUDO apt-get install -y "$p" || warn "  couldn't install $p"; done
            fi ;;
        dnf)
            pkgs=(gcc gcc-c++ make git curl wget pkgconf-pkg-config yasm nasm libwebp-devel)
            if ! $SUDO dnf install -y "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO dnf install -y "$p" || warn "  couldn't install $p"; done
            fi ;;
        yum)
            pkgs=(gcc gcc-c++ make git curl wget pkgconfig yasm nasm libwebp-devel)
            if ! $SUDO yum install -y "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO yum install -y "$p" || warn "  couldn't install $p"; done
            fi ;;
        pacman)
            pkgs=(base-devel git curl wget pkgconf yasm nasm libwebp)
            if ! $SUDO pacman -Sy --noconfirm --needed "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO pacman -S --noconfirm --needed "$p" || warn "  couldn't install $p"; done
            fi ;;
        zypper)
            pkgs=(gcc gcc-c++ make git curl wget pkg-config yasm nasm libwebp-devel)
            if ! $SUDO zypper --non-interactive install "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO zypper --non-interactive install "$p" || warn "  couldn't install $p"; done
            fi ;;
        apk)
            pkgs=(build-base git curl wget pkgconf yasm nasm libwebp-dev)
            if ! $SUDO apk add --no-cache "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO apk add --no-cache "$p" || warn "  couldn't install $p"; done
            fi ;;
        xbps)
            pkgs=(base-devel git curl wget pkgconf yasm nasm libwebp-devel)
            if ! $SUDO xbps-install -Sy "${pkgs[@]}"; then
                warn "batch install failed; retrying package-by-package"
                for p in "${pkgs[@]}"; do $SUDO xbps-install -Sy "$p" || warn "  couldn't install $p"; done
            fi ;;
    esac
}

# libwebp is a hard requirement for the miku codec specifically (not just a
# generic "nice to have" FFmpeg dep), so it gets its own fallback: build it
# from upstream source if the OS package didn't provide a usable one.
ensure_libwebp() {
    if pkg-config --exists libwebp 2>/dev/null; then
        ok "libwebp found via pkg-config ($(pkg-config --modversion libwebp))"
        return 0
    fi
    warn "libwebp not found via pkg-config after the dependency step."
    warn "falling back to building libwebp from source -- this is required, not optional, for the miku codec."

    local wd
    wd="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$wd'" RETURN

    if command -v git >/dev/null 2>&1 && git clone --depth 1 https://github.com/webmproject/libwebp.git "$wd/libwebp"; then
        :
    elif command -v curl >/dev/null 2>&1; then
        mkdir -p "$wd/libwebp"
        curl -fL https://github.com/webmproject/libwebp/archive/refs/heads/main.tar.gz | tar -xz -C "$wd/libwebp" --strip-components=1
    elif command -v wget >/dev/null 2>&1; then
        mkdir -p "$wd/libwebp"
        wget -qO- https://github.com/webmproject/libwebp/archive/refs/heads/main.tar.gz | tar -xz -C "$wd/libwebp" --strip-components=1
    else
        die "no git, curl, or wget available -- can't fetch libwebp as a fallback. Install one of those and re-run."
    fi

    (
        cd "$wd/libwebp"
        if [ -x autogen.sh ]; then
            ./autogen.sh
            ./configure --enable-shared --disable-static >/dev/null
            make -j"$JOBS"
            $SUDO make install
        elif [ -f CMakeLists.txt ] && command -v cmake >/dev/null 2>&1; then
            mkdir -p build && cd build
            cmake -DCMAKE_BUILD_TYPE=Release ..
            make -j"$JOBS"
            $SUDO make install
        else
            exit 1
        fi
    ) || die "building libwebp from source failed (no autogen.sh/CMakeLists.txt found, cmake missing, or the build itself failed)"

    $SUDO ldconfig 2>/dev/null || true

    if ! pkg-config --exists libwebp 2>/dev/null; then
        die "still can't find libwebp via pkg-config after building it from source. It may have installed under /usr/local -- check PKG_CONFIG_PATH includes /usr/local/lib/pkgconfig."
    fi
    ok "libwebp built and installed from source ($(pkg-config --modversion libwebp))"
}

# ---------------------------------------------------------------------------
# FFmpeg source: git clone -> full clone+checkout -> tarball, in that order
# ---------------------------------------------------------------------------
fetch_ffmpeg() {
    if [ "$DO_CLEAN" -eq 1 ] && [ -d "$BUILD_DIR" ]; then
        log "clean requested: removing existing $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    if [ -d "$BUILD_DIR/.git" ]; then
        log "reusing existing checkout at $BUILD_DIR, updating to $FFMPEG_REF"
        if ! ( cd "$BUILD_DIR" \
               && git remote set-url origin "$FFMPEG_REPO" \
               && git fetch --depth 1 origin "$FFMPEG_REF" \
               && git checkout -f FETCH_HEAD ); then
            warn "updating the existing checkout failed; falling back to a fresh clone"
            rm -rf "$BUILD_DIR"
        fi
    fi
    [ -d "$BUILD_DIR/.git" ] && return 0

    log "cloning FFmpeg ($FFMPEG_REF) from $FFMPEG_REPO"
    if command -v git >/dev/null 2>&1 \
       && git clone --depth 1 --branch "$FFMPEG_REF" "$FFMPEG_REPO" "$BUILD_DIR" 2>/dev/null; then
        ok "cloned via git (shallow, by branch/tag)"
        return 0
    fi

    warn "shallow clone of ref '$FFMPEG_REF' as a branch/tag failed (or git isn't available)."
    warn "trying a full clone + checkout instead -- slower, but works for arbitrary commit hashes too."
    rm -rf "$BUILD_DIR"
    if command -v git >/dev/null 2>&1 && git clone "$FFMPEG_REPO" "$BUILD_DIR" 2>/dev/null; then
        if ( cd "$BUILD_DIR" && git checkout "$FFMPEG_REF" ); then
            ok "cloned and checked out $FFMPEG_REF via git"
            return 0
        fi
        warn "git checkout of '$FFMPEG_REF' failed even after a full clone."
        rm -rf "$BUILD_DIR"
    fi

    warn "falling back to a source tarball download (no git required)."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    local tarball_urls=(
        "https://github.com/FFmpeg/FFmpeg/archive/refs/heads/$FFMPEG_REF.tar.gz"
        "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/$FFMPEG_REF.tar.gz"
        "https://github.com/FFmpeg/FFmpeg/archive/$FFMPEG_REF.tar.gz"
    )
    local url fetched=0
    for url in "${tarball_urls[@]}"; do
        log "trying $url"
        if command -v curl >/dev/null 2>&1 && curl -fL "$url" 2>/dev/null | tar -xz -C "$BUILD_DIR" --strip-components=1; then
            fetched=1; break
        elif command -v wget >/dev/null 2>&1 && wget -qO- "$url" 2>/dev/null | tar -xz -C "$BUILD_DIR" --strip-components=1; then
            fetched=1; break
        fi
    done

    if [ "$fetched" -ne 1 ]; then
        die "couldn't get FFmpeg source via git or a tarball for ref '$FFMPEG_REF'.
Check your network, or try a different --ref (a known release tag like n7.1 is a safe, stable bet)."
    fi
    ok "fetched FFmpeg source via tarball"
}

# ---------------------------------------------------------------------------
# Patch application + file placement
# ---------------------------------------------------------------------------
apply_miku_patch() {
    cd "$BUILD_DIR"

    if grep -q "AV_CODEC_ID_MIKU_DELTA" libavcodec/codec_id.h 2>/dev/null; then
        log "miku patch already applied in this checkout, skipping re-apply"
    else
        log "applying miku-support.patch"
        if git apply --check "$SOURCE_DIR/miku-support.patch" 2>/dev/null; then
            git apply "$SOURCE_DIR/miku-support.patch"
            ok "patch applied via git apply"
        elif patch -p1 --dry-run --silent < "$SOURCE_DIR/miku-support.patch" >/dev/null 2>&1; then
            patch -p1 < "$SOURCE_DIR/miku-support.patch"
            ok "patch applied via patch -p1"
        elif patch -p1 --fuzz=3 --dry-run --silent < "$SOURCE_DIR/miku-support.patch" >/dev/null 2>&1; then
            warn "clean apply failed; using --fuzz=3 (upstream has likely drifted since this patch was made)"
            patch -p1 --fuzz=3 < "$SOURCE_DIR/miku-support.patch"
            warn "applied with fuzz -- worth diffing the touched files against miku-support.patch by hand to sanity-check"
        else
            die "miku-support.patch doesn't apply against $FFMPEG_REF, even with fuzz.
This usually means upstream FFmpeg has drifted too far from whatever commit the patch was written against.
Try an older --ref (a specific release tag, e.g. --ref n7.1), or resolve it by hand:
  cd $BUILD_DIR && patch -p1 < $SOURCE_DIR/miku-support.patch
and fix up whatever .rej files that leaves behind."
        fi
    fi

    log "copying miku source files into place"
    cp -v "$SOURCE_DIR/mikuenc.c"      libavcodec/mikuenc.c
    cp -v "$SOURCE_DIR/mikudeltadec.c" libavcodec/mikudeltadec.c
    cp -v "$SOURCE_DIR/mikudec.c"      libavformat/mikudec.c
    cp -v "$SOURCE_DIR/mikumux.c"      libavformat/mikumux.c

    local all_ok=1 marker file needle
    for marker in \
        "libavcodec/Makefile:CONFIG_MIKU_ENCODER" \
        "libavcodec/allcodecs.c:ff_miku_encoder" \
        "libavcodec/codec_id.h:AV_CODEC_ID_MIKU_DELTA" \
        "libavcodec/packet.h:AV_PKT_DATA_MIKU_IS_DELTA" \
        "libavformat/Makefile:CONFIG_MIKU_MUXER" \
        "libavformat/allformats.c:ff_miku_muxer" \
    ; do
        file="${marker%%:*}"; needle="${marker##*:}"
        if ! grep -q "$needle" "$file" 2>/dev/null; then
            err "expected '$needle' in $file after patching, but it's not there."
            all_ok=0
        fi
    done
    [ "$all_ok" -eq 1 ] || die "patch verification failed -- see above. Without this, the build would silently not include miku support."
    ok "patch verified: all expected markers present"
}

# ---------------------------------------------------------------------------
# Configure + build
# ---------------------------------------------------------------------------
configure_and_build() {
    cd "$BUILD_DIR"

    local args=(--enable-libwebp --disable-doc)
    [ -n "$PREFIX" ] && args+=(--prefix="$PREFIX")
    if [ -n "$EXTRA_CONFIGURE_ARGS" ]; then
        # shellcheck disable=SC2206
        local extra=($EXTRA_CONFIGURE_ARGS)
        args+=("${extra[@]}")
    fi

    # Try with explicit component names first (most reliable if right);
    # fall back to plain --enable-libwebp (relies on FFmpeg's normal
    # default-enable-everything-with-satisfied-deps behavior) if the
    # explicit names turn out wrong for this FFmpeg version and configure
    # rejects them outright.
    local explicit_args=("${args[@]}" --enable-encoder=miku --enable-decoder=miku_delta --enable-muxer=miku --enable-demuxer=miku)
    log "configuring: ./configure ${explicit_args[*]}"
    if ./configure "${explicit_args[@]}"; then
        ok "configure succeeded with explicit miku component names"
    else
        warn "configure failed with explicit component names (they may not match this FFmpeg version's naming)."
        warn "falling back to a plain --enable-libwebp configure."
        log "configuring: ./configure ${args[*]}"
        if ! ./configure "${args[@]}"; then
            err "configure failed both with and without explicit component names."
            err "tail of ffbuild/config.log:"
            tail -n 40 ffbuild/config.log 2>/dev/null >&2 || true
            die "configure failed, see above"
        fi
    fi

    if ! grep -q "^#define CONFIG_MIKU_ENCODER 1" config.h 2>/dev/null; then
        warn "configure ran, but CONFIG_MIKU_ENCODER isn't enabled in config.h."
        warn "this almost always means libwebp wasn't detected at configure time -- check the 'libwebp' line in ffbuild/config.log."
        warn "continuing anyway so a real, specific error surfaces at build time instead of a silently incomplete build."
    fi

    log "building with $JOBS parallel job(s) -- this takes a while"
    if ! make -j"$JOBS"; then
        warn "parallel build failed; retrying with -j1 for a clean, attributable error"
        make -j1 || die "build failed -- see the error above for the real cause."
    fi
    ok "build finished"
}

# ---------------------------------------------------------------------------
# Install + verify
# ---------------------------------------------------------------------------
install_ffmpeg() {
    if [ "$DO_INSTALL" -eq 0 ]; then
        log "skipping install (--no-install). Binaries are at $BUILD_DIR/ffmpeg and $BUILD_DIR/ffprobe."
        return 0
    fi
    cd "$BUILD_DIR"
    log "installing (sudo make install)"
    if ! $SUDO make install; then
        die "make install failed. If you don't have sudo/root, re-run with --no-install and either
run 'make install' yourself as root, or use the binaries straight from $BUILD_DIR."
    fi
    $SUDO ldconfig 2>/dev/null || true
    ok "installed"
}

verify_install() {
    [ "$DO_INSTALL" -eq 1 ] || return 0

    local ffmpeg_bin
    ffmpeg_bin="$(command -v ffmpeg || true)"
    if [ -z "$ffmpeg_bin" ]; then
        warn "installed, but 'ffmpeg' isn't on PATH in this shell (open a new shell, or check --prefix is on PATH)."
        return 0
    fi

    log "verifying: $ffmpeg_bin"
    local found=1
    "$ffmpeg_bin" -hide_banner -muxers   2>/dev/null | grep -qi ' miku *' || { warn "miku muxer not listed";   found=0; }
    "$ffmpeg_bin" -hide_banner -demuxers 2>/dev/null | grep -qi ' miku *' || { warn "miku demuxer not listed"; found=0; }
    "$ffmpeg_bin" -hide_banner -encoders 2>/dev/null | grep -qi 'miku'    || { warn "miku encoder not listed"; found=0; }
    "$ffmpeg_bin" -hide_banner -decoders 2>/dev/null | grep -qi 'miku'    || { warn "miku_delta decoder not listed"; found=0; }

    if [ "$found" -eq 1 ]; then
        ok "miku muxer/demuxer/encoder/decoder all present -- $("$ffmpeg_bin" -version | head -n1)"
    else
        warn "install finished but one or more miku components aren't showing up -- see warnings above, and check ffbuild/config.log in $BUILD_DIR."
    fi
}

# ---------------------------------------------------------------------------
main() {
    preflight
    install_deps
    ensure_libwebp
    fetch_ffmpeg
    apply_miku_patch
    configure_and_build
    install_ffmpeg
    verify_install

    {
        echo "last built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "ffmpeg ref requested: $FFMPEG_REF"
        (cd "$BUILD_DIR" && git rev-parse HEAD 2>/dev/null) || echo "commit: unknown (non-git checkout)"
    } > "$BUILD_DIR/.miku-build-info" 2>/dev/null || true

    ok "done."
    log "re-run this script any time to rebuild against a newer FFmpeg -- it reuses $BUILD_DIR."
    log "use --clean for a fresh checkout, --ref <tag> to pin a specific FFmpeg version, --help for all options."
}

main "$@"
