#!/usr/bin/env bash
#miku-help-start
# miku-desktop-install.sh -- full desktop install for .miku support:
#   1. builds FFmpeg with .miku support (via build-miku-ffmpeg.sh) and
#      installs it under your home directory, set first on PATH
#   2. registers the video/x-miku MIME type, a thumbnailer, and a default
#      player, so .miku files show up as videos with thumbnails and
#      double-click to play
#   3. optionally builds mpv from source against *this* FFmpeg, so mpv --
#      and anything using libmpv, like Celluloid -- can decode .miku too
#
# Run from the directory containing mikuenc.c, mikudeltadec.c, mikudec.c,
# mikumux.c, miku-support.patch, and build-miku-ffmpeg.sh, e.g.:
#   cd /path/to/miku-source && ./miku-desktop-install.sh
#
# Nothing here touches system directories or needs sudo -- everything
# installs under $HOME. Safe to re-run; each step skips or updates
# cleanly rather than duplicating.
#
# Options:
#   --prefix <path>       Where FFmpeg (and mpv, if built) install to.
#                          Default: $HOME/.local/miku
#   --source-dir <path>   Directory with the miku*.c/.patch files and
#                          build-miku-ffmpeg.sh. Default: this script's dir.
#   --ref <git-ref>       FFmpeg ref to build. Default: master.
#   --ffmpeg-build <full|basic|ask>
#                          full: a normal FFmpeg build (every codec, format,
#                          and filter upstream enables by default, plus
#                          hardware acceleration and network streaming) --
#                          matches or exceeds your system ffmpeg, capability
#                          for capability, with .miku added on top.
#                          basic: the same full codec/format/filter coverage,
#                          but without hwaccel, network protocols, or
#                          capture devices -- a real, complete ffmpeg for
#                          file-to-file conversion, just leaner and faster
#                          to build, and with fewer optional platform
#                          dependencies that can fail to build on an
#                          unusual system. Default: ask.
#                          (The small static ffmpeg the thumbnailer uses is
#                          separate from this choice and always minimal --
#                          see build_thumb_ffmpeg's own comment for why.)
#   --jobs <N>             Parallel build jobs. Default: nproc.
#   --player <yes|no|ask>  Build mpv + wire up a player. Default: ask.
#   --no-deps              Skip the OS package-manager dependency step.
#   -h, --help              Show this help and exit.
#
# What actually happens to your system, precisely:
#   - Files written under $PREFIX (FFmpeg, optionally mpv/libmpv)
#   - A PATH line added to your shell rc file(s) (~/.bashrc, ~/.zshrc,
#     ~/.config/fish/config.fish, ~/.profile -- whichever exist)
#   - ~/.local/share/mime/packages/x-miku.xml (MIME type)
#   - ~/.local/share/thumbnailers/miku.thumbnailer + a small helper script
#   - ~/.local/share/applications/*.desktop (player entry) +
#     an xdg-mime default association for video/x-miku
#   - Packages installed via your OS package manager (build tooling only,
#     the --no-deps flag skips this if you'd rather manage it yourself)
#
# Known, honest limitations (see the final summary this script prints):
#   - GNOME/Nautilus thumbnails: should work. KDE/Dolphin on Plasma 6:
#     uncertain -- there's an open upstream regression affecting custom
#     .thumbnailer files as of this writing. Not something a script can
#     fix; noted so you're not left guessing if thumbnails don't appear.
#   - File-manager "Properties" panels showing duration/resolution/codec
#     needs a compiled indexer plugin (Tracker on GNOME, Baloo on KDE) --
#     genuinely out of scope for a portable script. mpv/Celluloid show
#     full metadata once the file is open, just not from a hover/properties
#     view without opening it.
#miku-help-end
set -Eeuo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "error: this script needs bash (run it as: bash $0)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'; C_MAGENTA=$'\033[35m'
    IS_TTY=1
else
    C_RESET=""; C_BOLD=""; C_DIM=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_MAGENTA=""
    IS_TTY=0
fi
SPIN_FRAMES='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

say()  { printf '%s\n' "$*"; }
head_() { printf '\n%s%s==>%s %s%s%s\n' "$C_BOLD" "$C_MAGENTA" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
warn() { printf '%s!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()  { printf '%s✗%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
die()  { err "$*"; exit 1; }

LOGDIR="$(mktemp -d)"
STEP_FAILED=0

# Runs "$@" with output captured to a per-step logfile; shows a spinner (or
# just the label, if not an interactive terminal) while it runs, then a
# green check or red cross. Returns the command's own exit status.
run_step() {
    local desc="$1"; shift
    local logfile="$LOGDIR/$(printf '%s' "$desc" | tr -c 'a-zA-Z0-9' '_').log"

    "$@" > "$logfile" 2>&1 &
    local pid=$!

    if [ "$IS_TTY" -eq 1 ]; then
        local i=0 len=${#SPIN_FRAMES}
        tput civis 2>/dev/null || true
        while kill -0 "$pid" 2>/dev/null; do
            printf "\r${C_CYAN}%s${C_RESET} %s" "${SPIN_FRAMES:$((i % len)):1}" "$desc"
            i=$((i + 1)); sleep 0.08
        done
        tput cnorm 2>/dev/null || true
        printf "\r\033[K"
    else
        printf '%s ... ' "$desc"
    fi

    local status=0
    wait "$pid" || status=$?
    if [ "$status" -eq 0 ]; then
        printf "${C_GREEN}✓${C_RESET} %s\n" "$desc"
    else
        printf "${C_RED}✗${C_RESET} %s\n" "$desc"
        printf "  ${C_DIM}log: %s${C_RESET}\n" "$logfile"
        printf "  ${C_DIM}last few lines:${C_RESET}\n"
        tail -n 8 "$logfile" 2>/dev/null | sed 's/^/    /'
        STEP_FAILED=1
    fi
    return "$status"
}

# Same idea, but for a step you want to *require* -- exits the script on failure.
require_step() { run_step "$@" || die "required step failed, stopping here"; }

# ---------------------------------------------------------------------------
# Defaults / flags
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"
PREFIX="$HOME/.local/miku"
FFMPEG_REF="master"
JOBS="$(nproc 2>/dev/null || echo 2)"
PLAYER_MODE="ask"     # yes | no | ask
FFMPEG_BUILD_TYPE="ask"   # full | basic | ask
DO_DEPS=1

print_help() {
    sed -n '/^#miku-help-start$/,/^#miku-help-end$/p' "${BASH_SOURCE[0]}" \
        | sed -e '1d;$d' -e 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)       PREFIX="$2"; shift 2 ;;
        --source-dir)   SOURCE_DIR="$2"; shift 2 ;;
        --ref)          FFMPEG_REF="$2"; shift 2 ;;
        --jobs)         JOBS="$2"; shift 2 ;;
        --player)       PLAYER_MODE="$2"; shift 2 ;;
        --ffmpeg-build) FFMPEG_BUILD_TYPE="$2"; shift 2 ;;
        --no-deps)      DO_DEPS=0; shift ;;
        -h|--help)      print_help; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

BUILD_ROOT="$PREFIX/build"
THUMB_PREFIX="$PREFIX/thumb-static"
BIN_DIR="$PREFIX/bin"
MIME_DIR="$HOME/.local/share/mime/packages"
THUMBNAILER_DIR="$HOME/.local/share/thumbnailers"
APPS_DIR="$HOME/.local/share/applications"
HELPERS_DIR="$PREFIX/libexec"

REQUIRED_FILES=(mikuenc.c mikudeltadec.c mikudec.c mikumux.c miku-support.patch build-miku-ffmpeg.sh)

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
head_ "MIKU desktop install"
say "  source dir : $SOURCE_DIR"
say "  prefix     : $PREFIX"
say "  ffmpeg ref : $FFMPEG_REF"
say "  jobs       : $JOBS"

[ "$(uname -s)" = "Linux" ] || die "this script targets Linux desktops."

missing=()
for f in "${REQUIRED_FILES[@]}"; do
    [ -f "$SOURCE_DIR/$f" ] || missing+=("$f")
done
[ "${#missing[@]}" -eq 0 ] || die "missing file(s) in $SOURCE_DIR: ${missing[*]}
Run this from the directory with the miku source files and build-miku-ffmpeg.sh, or pass --source-dir."

PKG_MGR=""
if   command -v apt-get      >/dev/null 2>&1; then PKG_MGR=apt
elif command -v dnf          >/dev/null 2>&1; then PKG_MGR=dnf
elif command -v pacman       >/dev/null 2>&1; then PKG_MGR=pacman
elif command -v zypper       >/dev/null 2>&1; then PKG_MGR=zypper
elif command -v apk          >/dev/null 2>&1; then PKG_MGR=apk
fi
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    # Every later use of $SUDO happens inside run_step, which backgrounds
    # the command and redirects its output to a logfile -- if sudo needed
    # to prompt for a password at that point, the prompt would be invisible
    # and the step would just hang. Ask for it here instead, visibly,
    # before anything is backgrounded, then keep the credential cache warm
    # for the whole run (a full build can run well past sudo's normal
    # timeout) via a quiet refresher killed on exit.
    say "Some steps install packages via $SUDO -- you may be asked for your password now."
    sudo -v || die "sudo authentication failed"
    ( while true; do sudo -n true 2>/dev/null; sleep 60; done ) &
    SUDO_KEEPALIVE_PID=$!
    trap '[ -n "${SUDO_KEEPALIVE_PID:-}" ] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT
fi

mkdir -p "$PREFIX" "$BUILD_ROOT" "$BIN_DIR" "$MIME_DIR" "$THUMBNAILER_DIR" "$APPS_DIR" "$HELPERS_DIR"

# Ask anything we need to ask now, together, before any long-running
# (and backgrounded, spinner-driven) step starts -- so once the real work
# begins you don't have to sit and babysit it for a prompt partway through
# a build that can run well past an hour on --ffmpeg-build full.
if [ "$FFMPEG_BUILD_TYPE" = "ask" ]; then
    if [ "$IS_TTY" -eq 1 ]; then
        printf "%sFull FFmpeg (every codec/format, hwaccel, network streaming -- matches your\n"  "$C_BOLD"
        printf "system ffmpeg) or basic (same codec/format coverage, no hwaccel/network/\n"
        printf "devices -- leaner, and fewer optional deps that can fail to build)? [F/b] %s" "$C_RESET"
        read -r reply || reply="f"
        case "$reply" in [bB]*) FFMPEG_BUILD_TYPE="basic" ;; *) FFMPEG_BUILD_TYPE="full" ;; esac
    else
        FFMPEG_BUILD_TYPE="full"
    fi
fi
WANT_PLAYER=0
case "$PLAYER_MODE" in
    yes) WANT_PLAYER=1 ;;
    no)  WANT_PLAYER=0 ;;
    ask)
        if [ "$IS_TTY" -eq 1 ]; then
            printf "%sBuild mpv (and wire up Celluloid if present) so double-click just plays .miku files? [Y/n] %s" "$C_BOLD" "$C_RESET"
            read -r reply || reply="y"
            case "$reply" in [nN]*) WANT_PLAYER=0 ;; *) WANT_PLAYER=1 ;; esac
        else
            WANT_PLAYER=1
        fi ;;
esac
say ""
say "FFmpeg build : ${C_BOLD}$FFMPEG_BUILD_TYPE${C_RESET}"
say "Player setup : ${C_BOLD}$([ "$WANT_PLAYER" -eq 1 ] && echo "yes" || echo "no")${C_RESET}"
say "${C_DIM}(everything from here runs unattended)${C_RESET}"

# ---------------------------------------------------------------------------
# Dependencies (mpv's own deps, in addition to whatever build-miku-ffmpeg.sh
# installs for FFmpeg itself). Same "batch install, retry package-by-package
# on failure" resilience as that script, so one unavailable/renamed package
# on your distro doesn't block everything else.
# ---------------------------------------------------------------------------
install_deps() {
    [ "$DO_DEPS" -eq 1 ] || { say "skipping dependency install (--no-deps)"; return 0; }
    [ -n "$PKG_MGR" ] || { warn "no known package manager found; make sure build tools, meson, ninja, and mpv's usual deps are already installed"; return 0; }

    local pkgs=()
    case "$PKG_MGR" in
        apt)
            pkgs=(meson ninja-build pkg-config git curl desktop-file-utils shared-mime-info
                  libplacebo-dev libass-dev libvulkan-dev glslang-tools libshaderc-dev
                  libx11-dev libxext-dev libxrandr-dev libxpresent-dev libxss-dev
                  libwayland-dev wayland-protocols libxkbcommon-dev
                  libpulse-dev libasound2-dev)
            $SUDO apt-get update -y >/dev/null 2>&1 || true
            if ! $SUDO apt-get install -y "${pkgs[@]}" >/dev/null 2>&1; then
                for p in "${pkgs[@]}"; do $SUDO apt-get install -y "$p" >/dev/null 2>&1 || true; done
            fi ;;
        dnf)
            pkgs=(meson ninja-build pkgconf-pkg-config git curl desktop-file-utils shared-mime-info
                  libplacebo-devel libass-devel vulkan-loader-devel glslang shaderc
                  libX11-devel libXext-devel libXrandr-devel libXpresent-devel libXScrnSaver-devel
                  wayland-devel wayland-protocols-devel libxkbcommon-devel
                  pulseaudio-libs-devel alsa-lib-devel)
            if ! $SUDO dnf install -y "${pkgs[@]}" >/dev/null 2>&1; then
                for p in "${pkgs[@]}"; do $SUDO dnf install -y "$p" >/dev/null 2>&1 || true; done
            fi ;;
        pacman)
            pkgs=(meson ninja pkgconf git curl desktop-file-utils shared-mime-info
                  libplacebo libass vulkan-icd-loader glslang shaderc
                  libx11 libxext libxrandr libxpresent libxss
                  wayland wayland-protocols libxkbcommon
                  libpulse alsa-lib)
            if ! $SUDO pacman -Sy --noconfirm --needed "${pkgs[@]}" >/dev/null 2>&1; then
                for p in "${pkgs[@]}"; do $SUDO pacman -S --noconfirm --needed "$p" >/dev/null 2>&1 || true; done
            fi ;;
        zypper)
            pkgs=(meson ninja pkg-config git curl desktop-file-utils shared-mime-info
                  libplacebo-devel libass-devel vulkan-devel glslang shaderc
                  libX11-devel libXext-devel libXrandr-devel libXss-devel
                  wayland-devel wayland-protocols-devel libxkbcommon-devel
                  libpulse-devel alsa-devel)
            if ! $SUDO zypper --non-interactive install "${pkgs[@]}" >/dev/null 2>&1; then
                for p in "${pkgs[@]}"; do $SUDO zypper --non-interactive install "$p" >/dev/null 2>&1 || true; done
            fi ;;
        apk)
            pkgs=(meson ninja pkgconf git curl desktop-file-utils shared-mime-info
                  libplacebo-dev libass-dev vulkan-loader-dev glslang shaderc
                  libx11-dev libxext-dev libxrandr-dev
                  wayland-dev wayland-protocols libxkbcommon-dev
                  pulseaudio-dev alsa-lib-dev)
            if ! $SUDO apk add --no-cache "${pkgs[@]}" >/dev/null 2>&1; then
                for p in "${pkgs[@]}"; do $SUDO apk add --no-cache "$p" >/dev/null 2>&1 || true; done
            fi ;;
    esac
}

# ---------------------------------------------------------------------------
# Step 1: FFmpeg with .miku support, installed under $PREFIX
# ---------------------------------------------------------------------------
FFMPEG_REPO="https://github.com/FFmpeg/FFmpeg.git"

build_ffmpeg() {
    local bdir="$BUILD_ROOT/ffmpeg"
    # Deliberately NOT passing --no-deps here: build-miku-ffmpeg.sh's own
    # dependency list (compiler, nasm/yasm, libwebp) is what FFmpeg itself
    # needs and this script's own install_deps() doesn't cover -- that one
    # is purely additive, for mpv and desktop-integration tools on top.
    # Some overlap (git, curl, pkg-config) is harmless.
    local extra_args="--enable-shared --disable-static"
    # "basic" keeps every codec, container format, and filter upstream
    # enables by default -- this is still a real, complete ffmpeg for
    # file-to-file conversion -- and only drops three *categories* that
    # aren't needed for that: hardware acceleration, network protocols,
    # and capture/playback devices. Each of those also pulls in its own
    # optional platform dependencies (CUDA/VAAPI/ALSA headers, etc.) that
    # are a real, independent source of build failures on an unusual
    # system, so "basic" is also the more reliably-buildable of the two,
    # not just the smaller one.
    if [ "$FFMPEG_BUILD_TYPE" = "basic" ]; then
        extra_args="$extra_args --disable-hwaccels --disable-network --disable-devices"
    fi
    local deps_flag=""
    [ "$DO_DEPS" -eq 1 ] || deps_flag="--no-deps"
    bash "$SOURCE_DIR/build-miku-ffmpeg.sh" \
        --ref "$FFMPEG_REF" --source-dir "$SOURCE_DIR" --build-dir "$bdir" \
        --jobs "$JOBS" --prefix "$PREFIX" \
        --extra-configure-args "$extra_args" \
        --no-install $deps_flag
    # --no-install: build-miku-ffmpeg.sh's own installer assumes it may need
    # sudo; we install ourselves below, without sudo, since $PREFIX is
    # entirely user-owned.
    ( cd "$bdir" && make install )
}

# A second, minimal, *statically* linked ffmpeg, used only by the
# thumbnailer. GNOME/Nautilus runs thumbnailers inside a bubblewrap sandbox
# that only exposes /usr, /lib and /lib64 by default -- a dynamically
# linked binary depending on libraries under $PREFIX (outside those paths)
# will fail to even start there, silently, with no thumbnail and no obvious
# error. A static binary has no such dependency: `ldd` on it shows only
# libc/libm/libz, all of which the sandbox already exposes. Verified this
# distinction directly rather than assuming it.
build_thumb_ffmpeg() {
    local bdir="$BUILD_ROOT/ffmpeg-thumb"
    mkdir -p "$bdir"
    if [ ! -d "$bdir/.git" ]; then
        cp -r "$BUILD_ROOT/ffmpeg/." "$bdir/" 2>/dev/null || \
            git clone --depth 1 "$FFMPEG_REPO" "$bdir"
    fi
    (
        cd "$bdir"
        ./configure --prefix="$THUMB_PREFIX" --disable-everything \
            --enable-static --disable-shared --pkg-config-flags=--static \
            --enable-libwebp \
            --enable-decoder=miku_delta,webp --enable-demuxer=miku \
            --enable-filter=thumbnail,scale --enable-muxer=image2 \
            --enable-encoder=png --enable-protocol=file \
            --enable-avcodec --enable-avformat --enable-avutil --enable-swscale \
            --enable-ffmpeg --disable-ffprobe --disable-doc --disable-x86asm
        make -j"$JOBS"
        make install
    )
}

# ---------------------------------------------------------------------------
# Step 2: PATH
# ---------------------------------------------------------------------------
setup_path() {
    local marker="# added by miku-desktop-install.sh"
    local line="export PATH=\"$BIN_DIR:\$PATH\" $marker"
    local rc found=0
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        [ -f "$rc" ] || continue
        found=1
        grep -qF "$marker" "$rc" 2>/dev/null || printf '\n%s\n' "$line" >> "$rc"
    done
    local fish="$HOME/.config/fish/config.fish"
    if [ -f "$fish" ]; then
        found=1
        grep -qF "$marker" "$fish" 2>/dev/null || \
            printf '\nfish_add_path %s %s\n' "$BIN_DIR" "$marker" >> "$fish"
    fi
    [ "$found" -eq 1 ] || printf '\n%s\n' "$line" >> "$HOME/.profile"
    export PATH="$BIN_DIR:$PATH"
}

# ---------------------------------------------------------------------------
# Step 3: MIME type
# ---------------------------------------------------------------------------
setup_mime() {
    cat > "$MIME_DIR/x-miku.xml" << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="video/x-miku">
    <comment>MIKU Lossless Video</comment>
    <glob pattern="*.miku"/>
    <magic priority="60">
      <match type="string" value="MIKU" offset="0"/>
    </magic>
  </mime-type>
</mime-info>
XMLEOF
    if command -v update-mime-database >/dev/null 2>&1; then
        update-mime-database "$HOME/.local/share/mime" >/dev/null 2>&1
    else
        warn "update-mime-database not found -- install shared-mime-info to complete MIME registration"
    fi
}

# ---------------------------------------------------------------------------
# Step 4: thumbnailer (freedesktop spec -- GNOME/Nemo/Thunar via Tumbler
# generally honor this directly; see the final summary re: KDE/Plasma 6)
# ---------------------------------------------------------------------------
setup_thumbnailer() {
    local script="$HELPERS_DIR/miku-thumbnail.sh"
    cat > "$script" << SCRIPTEOF
#!/usr/bin/env bash
# Generates a thumbnail PNG from a .miku file: \$1=input \$2=output \$3=size
exec "$THUMB_PREFIX/bin/ffmpeg" -y -loglevel error -i "\$1" \\
    -vf "thumbnail,scale=\$3:-1" -frames:v 1 -f image2 "\$2"
SCRIPTEOF
    chmod +x "$script"

    cat > "$THUMBNAILER_DIR/miku.thumbnailer" << THUMBEOF
[Thumbnailer Entry]
TryExec=$THUMB_PREFIX/bin/ffmpeg
Exec=$script %i %o %s
MimeType=video/x-miku;
THUMBEOF
}

# ---------------------------------------------------------------------------
# Step 5: default player + xdg-mime association
# ---------------------------------------------------------------------------
write_desktop_entry() {
    # $1=id (no .desktop) $2=Name $3=Exec line $4=Icon
    cat > "$APPS_DIR/$1.desktop" << DESKEOF
[Desktop Entry]
Type=Application
Name=$2
Exec=$3 %f
Icon=$4
Terminal=false
MimeType=video/x-miku;
Categories=AudioVideo;Player;
DESKEOF
    command -v update-desktop-database >/dev/null 2>&1 && \
        update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
}

set_default_player() {
    local desktop_id="$1"
    command -v xdg-mime >/dev/null 2>&1 || { warn "xdg-mime not found -- can't set a default player automatically"; return 0; }
    xdg-mime default "$desktop_id.desktop" video/x-miku >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# Step 6 (optional): mpv, built against *our* ffmpeg
# ---------------------------------------------------------------------------
build_mpv() {
    local bdir="$BUILD_ROOT/mpv"
    if [ ! -d "$bdir/.git" ]; then
        git clone --depth 1 https://github.com/mpv-player/mpv.git "$bdir"
    fi
    local tag
    tag="$(git ls-remote --tags --refs https://github.com/mpv-player/mpv.git 'v0.*' \
           | awk '{print $2}' | sed 's#refs/tags/##' | sort -V | tail -1)"
    [ -n "$tag" ] || tag="v0.40.0"
    ( cd "$bdir" && git fetch --depth 1 origin "tag" "$tag" && git checkout -f "$tag" )

    (
        cd "$bdir"
        rm -rf build
        export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
        meson setup build -Dlibmpv=true -Dcplayer=true --buildtype=release \
            -Dc_link_args="-Wl,-rpath,$PREFIX/lib" -Dcpp_link_args="-Wl,-rpath,$PREFIX/lib"
        ninja -C build -j"$JOBS"
        ninja -C build install
    )
}

# ---------------------------------------------------------------------------
# Step 7 (optional): Celluloid, wired to use our libmpv
# ---------------------------------------------------------------------------
setup_celluloid() {
    local celluloid_bin=""
    celluloid_bin="$(command -v celluloid 2>/dev/null || true)"

    if [ -z "$celluloid_bin" ]; then
        case "$PKG_MGR" in
            apt)    $SUDO apt-get install -y celluloid >/dev/null 2>&1 || true ;;
            dnf)    $SUDO dnf install -y celluloid >/dev/null 2>&1 || true ;;
            pacman) $SUDO pacman -S --noconfirm --needed celluloid >/dev/null 2>&1 || true ;;
            zypper) $SUDO zypper --non-interactive install celluloid >/dev/null 2>&1 || true ;;
            apk)    $SUDO apk add --no-cache celluloid >/dev/null 2>&1 || true ;;
        esac
        celluloid_bin="$(command -v celluloid 2>/dev/null || true)"
    fi
    [ -n "$celluloid_bin" ] || return 1

    # A wrapper, not a rebuild of Celluloid itself: Celluloid links the
    # *system* libmpv, which can't decode .miku. Rebuilding Celluloid from
    # source against our libmpv would need its own GTK4 toolchain on top of
    # everything else here, so instead: launch the existing Celluloid with
    # our lib directory first on the loader's search path, *only* for this
    # one wrapper .desktop entry -- every other way of opening Celluloid
    # (app menu, other video files) is completely untouched and keeps using
    # the system libmpv exactly as before. mpv's client API has stayed
    # backward compatible for years, so this is low-risk, not zero-risk --
    # noted plainly in the final summary rather than glossed over.
    local wrapper="$HELPERS_DIR/celluloid-miku.sh"
    cat > "$wrapper" << WRAPEOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$PREFIX/lib:\${LD_LIBRARY_PATH:-}"
exec "$celluloid_bin" "\$@"
WRAPEOF
    chmod +x "$wrapper"
    write_desktop_entry "celluloid-miku" "Celluloid (MIKU)" "$wrapper" "celluloid"
    # run_step backgrounds whatever it calls (to animate the spinner), which
    # means this function's own stdout and any variables it sets never make
    # it back to the calling shell -- only its exit status does. So: write
    # the result somewhere the caller can read *after* run_step returns,
    # rather than trying to "return" it directly.
    echo "celluloid-miku" > "$LOGDIR/.player_id"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
head_ "Installing OS build dependencies"
run_step "Installing OS build dependencies" install_deps

head_ "Building FFmpeg with .miku support"
require_step "Pulling latest FFmpeg and applying the .miku patch, then compiling MIKU" build_ffmpeg
require_step "Building a small static FFmpeg for safe thumbnailing" build_thumb_ffmpeg
run_step "Adding $BIN_DIR to your PATH" setup_path

head_ "File manager integration"
run_step "Registering the video/x-miku file type" setup_mime
run_step "Setting up thumbnail generation" setup_thumbnailer

DEFAULT_DESKTOP_ID="mpv"
PLAYER_BUILT=0
# WANT_PLAYER was already resolved during the upfront prompts above.

if [ "$WANT_PLAYER" -eq 1 ]; then
    head_ "Player integration"
    if run_step "Building mpv against your new FFmpeg" build_mpv; then
        PLAYER_BUILT=1
        write_desktop_entry "mpv-miku" "mpv (MIKU)" "$PREFIX/bin/mpv" "mpv"
        DEFAULT_DESKTOP_ID="mpv-miku"
        if run_step "Wiring up Celluloid" setup_celluloid && [ -f "$LOGDIR/.player_id" ]; then
            DEFAULT_DESKTOP_ID="$(cat "$LOGDIR/.player_id")"
        fi
    fi
fi
run_step "Setting the default .miku player" set_default_player "$DEFAULT_DESKTOP_ID"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
head_ "Done"
say "FFmpeg with .miku support (${FFMPEG_BUILD_TYPE}): ${C_GREEN}$PREFIX/bin/ffmpeg${C_RESET}"
say "  (open a new terminal, or 'source ~/.bashrc', for PATH to take effect)"
if [ "$PLAYER_BUILT" -eq 1 ]; then
    say "Default .miku player: ${C_GREEN}$DEFAULT_DESKTOP_ID${C_RESET}"
    say "Double-click a .miku file in your file manager to try it."
else
    say "Player integration skipped -- FFmpeg itself is fully installed and usable."
fi
say ""
say "${C_BOLD}Known limitations, honestly:${C_RESET}"
say "  - Thumbnails: registered the standard way, which GNOME/Nemo/Thunar"
say "    read directly. KDE Plasma 6 has an open upstream bug affecting"
say "    custom .thumbnailer files -- if Dolphin doesn't show thumbnails,"
say "    that's very likely why, not something this script got wrong."
say "  - File manager 'Properties' panels showing duration/codec/resolution"
say "    needs a compiled Tracker/Baloo plugin -- out of scope here. Open"
say "    the file in mpv/Celluloid for full metadata."
if [ "$STEP_FAILED" -eq 1 ]; then
    say ""
    warn "one or more steps failed -- see the logs pointed to above."
fi
say ""
say "Logs for this run: $LOGDIR"
