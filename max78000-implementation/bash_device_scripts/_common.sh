# Shared helpers for device_*.sh and host_*.sh scripts.
# Source this from each script: . "$(dirname "$0")/_common.sh"

set -euo pipefail

# ---------- args -----------------------------------------------------------

usage() {
  echo "usage: $0 {baseline|improved|mininet|deeper|nascifarnet|ressimplenet}" >&2
  exit 1
}
[ "${1:-}" = "" ] && usage
VARIANT=$1
case "$VARIANT" in
  baseline|improved|mininet|deeper|nascifarnet|ressimplenet) ;;
  *) usage ;;
esac

# ---------- paths ----------------------------------------------------------

# Absolute path of the max78000-implementation directory (script lives in scripts/).
THIS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PC_DIR="$(cd "$THIS_DIR/../pc-implementation" && pwd)"

# AI root (default; override via env if your stack is elsewhere)
: "${AI:=$HOME/Desktop/project/max78000}"

# Force MAXIM_PATH to the correct location regardless of what's in ~/.zshrc.
# The MSDK lives under ai8x-synthesis/sdk in this project layout.
export MAXIM_PATH="$AI/ai8x-synthesis/sdk"
if [ ! -f "$MAXIM_PATH/Libraries/libs.mk" ]; then
  echo "ERROR: MAXIM_PATH=$MAXIM_PATH does not contain Libraries/libs.mk" >&2
  echo "       Adjust the export in bash_device_scripts/_common.sh to point at your MSDK clone." >&2
  exit 1
fi

# Ensure the ARM GNU toolchain is on PATH even when this script is invoked from
# a fresh shell that hasn't sourced ~/.zshrc. Try a few well-known locations.
for _arm_bin in \
    /usr/local/arm-gnu-toolchain-12.3.rel1/bin \
    /usr/local/arm-gnu-toolchain/bin \
    /Applications/ARM/bin \
    /opt/homebrew/opt/arm-none-eabi-gcc/bin; do
  if [ -x "$_arm_bin/arm-none-eabi-gcc" ]; then
    case ":$PATH:" in *":$_arm_bin:"*) ;; *) export PATH="$_arm_bin:$PATH" ;; esac
    break
  fi
done
unset _arm_bin
if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "ERROR: arm-none-eabi-gcc not found on PATH" >&2
  echo "       Edit bash_device_scripts/_common.sh and add your toolchain bin/ dir to the loop." >&2
  exit 1
fi

case "$VARIANT" in
  nascifarnet)  ARCH="ai85nascifarnet" ;;
  *)            ARCH="ai85net_cmsis_${VARIANT}" ;;
esac
SYNTH_DIR="$AI/ai8x-synthesis/synthed_net_${VARIANT}/${VARIANT}"

# ---------- helpers --------------------------------------------------------

detect_port() {
  local p
  for pattern in '/dev/cu.usbmodem*' '/dev/ttyACM*'; do
    p=$(ls $pattern 2>/dev/null | head -1 || true)
    [ -n "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

require_synth() {
  if [ ! -f "$SYNTH_DIR/cnn.c" ]; then
    echo "ERROR: no synthesized project at $SYNTH_DIR" >&2
    echo "       run: cd $THIS_DIR && bash train_max78000_models.sh $VARIANT" >&2
    exit 1
  fi
}

# Swap in a new main.c, keeping the auto-generated one as main.c.orig.
# Args:  $1 = absolute path to the .c file to install as main.c
#
# Safeguard: we ONLY save main.c.orig the first time, and only if the current
# main.c looks like the auto-generated one (i.e. it does NOT contain any of
# our harness's signature strings). This prevents accidentally cloning one of
# our broken harnesses as the "original".
swap_main_c() {
  local src=$1
  if [ ! -f "$src" ]; then
    echo "ERROR: source file $src not found" >&2
    exit 1
  fi
  if [ ! -f "$SYNTH_DIR/main.c.orig" ]; then
    if grep -q "profile_layers\.c\|inference_test_set\.c\|measure_inference\.c" \
        "$SYNTH_DIR/main.c" 2>/dev/null; then
      echo "ERROR: main.c already looks like one of our harnesses." >&2
      echo "       Refusing to save it as main.c.orig (would lose the auto-gen)." >&2
      echo "       Re-synthesize first:" >&2
      echo "         rm -rf $SYNTH_DIR/.." >&2
      echo "         cd $THIS_DIR && bash train_max78000_models.sh $VARIANT" >&2
      exit 1
    fi
    cp "$SYNTH_DIR/main.c" "$SYNTH_DIR/main.c.orig"
    echo "[swap_main_c] saved auto-generated main.c → main.c.orig"
  fi
  cp "$src" "$SYNTH_DIR/main.c"
  echo "[swap_main_c] installed $(basename "$src") as main.c"
  # Append load_input() from main.c.orig, but ONLY for harnesses that forward-declare it.
  # inference_test_set.c fills input_0[] from UART and must NOT get the sampledata version.
  if grep -qF 'void load_input(void)' "$src"; then
    if [ -f "$SYNTH_DIR/main.c.orig" ]; then
      {
        printf '\n/* ---- load_input() extracted from main.c.orig by swap_main_c ---- */\n'
        awk '
          /^static const uint32_t input_/ { p=1 }
          p { print }
          /^void load_input/ { saw_func=1 }
          /^}/ && p && saw_func { print ""; exit }
        ' "$SYNTH_DIR/main.c.orig"
      } >> "$SYNTH_DIR/main.c"
      echo "[swap_main_c] appended load_input() from main.c.orig"
    else
      echo "WARNING: main.c.orig not found — load_input() not appended." >&2
    fi
  else
    echo "[swap_main_c] harness manages its own input — load_input() not appended"
  fi
}

build_and_flash() {
  cd "$SYNTH_DIR"
  echo "[build] make distclean"
  make distclean >/dev/null
  echo "[build] make BOARD=FTHR_RevA -j"
  make BOARD=FTHR_RevA -j

  # The `make flash.openocd` target uses whatever `openocd` is on PATH (often
  # Homebrew's), which lacks the MAX78000 target config. Use the bundled
  # ai8x-synthesis OpenOCD wrapper which knows where to find it.
  local OPENOCD_DIR="$AI/ai8x-synthesis/openocd"
  local ELF="$SYNTH_DIR/build/max78000.elf"
  if [ ! -x "$OPENOCD_DIR/run-openocd-maxdap" ]; then
    echo "[flash] FALLBACK make flash.openocd (bundled wrapper missing)" >&2
    make TARGET=MAX78000 BOARD=FTHR_RevA flash.openocd
  else
    echo "[flash] bundled OpenOCD: program $ELF verify reset exit"
    ( cd "$OPENOCD_DIR" && \
      ./run-openocd-maxdap \
        -c "init" \
        -c "reset halt" \
        -c "program $ELF verify reset exit" \
        -c "shutdown" )
  fi
  echo "[done] flashed $VARIANT"
}
