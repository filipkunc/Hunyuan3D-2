#!/usr/bin/env bash
#
# Build Hunyuan3D-2 native extensions on Linux.
#
#   - differentiable_renderer  (CPU, pybind11)
#   - custom_rasterizer        (CUDA)
#
# The CUDA toolkit used by nvcc must have the same major version as the CUDA
# runtime bundled with PyTorch. A matching minor version is preferred.
#
# Usage:
#   ./build_extensions.sh
#   ./build_extensions.sh --cpu-only
#   ./build_extensions.sh --cuda-home /usr/local/cuda-12.8
#   ./build_extensions.sh --arch 12.0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
cd "$SCRIPT_DIR"

if [[ -t 1 ]]; then
    C='\033[36m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; N='\033[0m'
else
    C=''; G=''; Y=''; R=''; N=''
fi

info() { printf '%b==> %s%b\n' "$C" "$*" "$N"; }
ok()   { printf '%b    %s%b\n' "$G" "$*" "$N"; }
warn() { printf '%b    warning: %s%b\n' "$Y" "$*" "$N" >&2; }
err()  { printf '%berror: %s%b\n' "$R" "$*" "$N" >&2; exit 1; }

usage() {
    sed -n '2,16p' "$0"
}

CPU_ONLY=0
CUDA_HOME_OVERRIDE=""
ARCH_OVERRIDE=""

while (($#)); do
    case "$1" in
        --cpu-only)
            CPU_ONLY=1
            shift
            ;;
        --cuda-home)
            (($# >= 2)) || err "--cuda-home requires a path"
            CUDA_HOME_OVERRIDE="$2"
            shift 2
            ;;
        --arch)
            (($# >= 2)) || err "--arch requires a CUDA architecture such as 12.0"
            ARCH_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || err "this script currently supports Linux only"
[[ -x "$VENV_PY" ]] || err "venv Python not found at $VENV_PY; run 'uv sync' first"

# Build the CPU extension first. It does not depend on CUDA and remains useful
# even when a matching CUDA toolkit is not installed yet.
info "Building differentiable_renderer (CPU)"
(
    cd "$SCRIPT_DIR/hy3dgen/texgen/differentiable_renderer"
    "$VENV_PY" setup.py build_ext --inplace
) || err "differentiable_renderer build failed"

"$VENV_PY" -c \
    'from hy3dgen.texgen.differentiable_renderer import mesh_processor; print(f"    imported {mesh_processor.__file__}")'
ok "differentiable_renderer built and imported"

if ((CPU_ONLY)); then
    ok "CPU-only build complete"
    exit 0
fi

TORCH_CUDA="$("$VENV_PY" -c 'import torch; print(torch.version.cuda or "")')"
[[ -n "$TORCH_CUDA" ]] || err "PyTorch in the venv is not a CUDA build"
TORCH_MAJOR="${TORCH_CUDA%%.*}"
TORCH_MINOR="${TORCH_CUDA#*.}"
TORCH_MINOR="${TORCH_MINOR%%.*}"
TORCH_SERIES="$TORCH_MAJOR.$TORCH_MINOR"
info "PyTorch uses CUDA $TORCH_CUDA"

nvcc_release() {
    "$1/bin/nvcc" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' \
        | head -1
}

ENV_CUDA_HOME="${CUDA_HOME:-}"
SYSTEM_NVCC_HOME=""
if command -v nvcc >/dev/null 2>&1; then
    SYSTEM_NVCC_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

if [[ -n "$CUDA_HOME_OVERRIDE" ]]; then
    CUDA_CANDIDATES=("$CUDA_HOME_OVERRIDE")
else
    CUDA_CANDIDATES=(
        "$ENV_CUDA_HOME"
        "/usr/local/cuda-$TORCH_SERIES"
        "/usr/local/cuda-$TORCH_MAJOR"
        "/usr/local/cuda"
        "$SYSTEM_NVCC_HOME"
    )
fi

CUDA_DIR=""
COMPATIBLE_CUDA_DIR=""
COMPATIBLE_CUDA_RELEASE=""
AVAILABLE_TOOLKITS=()

for candidate in "${CUDA_CANDIDATES[@]}"; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    release="$(nvcc_release "$candidate")"
    [[ -n "$release" ]] || continue
    AVAILABLE_TOOLKITS+=("$candidate ($release)")

    if [[ "$release" == "$TORCH_SERIES" || "$release" == "$TORCH_SERIES".* ]]; then
        CUDA_DIR="$candidate"
        break
    fi

    if [[ "${release%%.*}" == "$TORCH_MAJOR" && -z "$COMPATIBLE_CUDA_DIR" ]]; then
        COMPATIBLE_CUDA_DIR="$candidate"
        COMPATIBLE_CUDA_RELEASE="$release"
    fi
done

if [[ -z "$CUDA_DIR" && -n "$COMPATIBLE_CUDA_DIR" ]]; then
    CUDA_DIR="$COMPATIBLE_CUDA_DIR"
    warn "using CUDA $COMPATIBLE_CUDA_RELEASE with PyTorch CUDA $TORCH_CUDA; an exact minor-version match is preferred"
fi

if [[ -z "$CUDA_DIR" ]]; then
    if ((${#AVAILABLE_TOOLKITS[@]})); then
        warn "detected toolkit(s): ${AVAILABLE_TOOLKITS[*]}"
    fi
    err "no CUDA $TORCH_MAJOR.x toolkit found for PyTorch CUDA $TORCH_CUDA; install one side-by-side or pass --cuda-home PATH"
fi

export CUDA_HOME="$CUDA_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_RELEASE="$(nvcc_release "$CUDA_HOME")"
ok "Using CUDA toolkit $CUDA_RELEASE at $CUDA_HOME"

# Prefer a supported side-by-side compiler when the distribution default is
# newer than the toolkit supports. CUDA 12.x supports GCC through 14, while
# CUDA 13.x supports GCC through 15.
CUDA_MAJOR="${CUDA_RELEASE%%.*}"
SUPPORTED_GCC_MAX=""
case "$CUDA_MAJOR" in
    12) SUPPORTED_GCC_MAX=14 ;;
    13) SUPPORTED_GCC_MAX=15 ;;
esac

if [[ -n "$SUPPORTED_GCC_MAX" ]]; then
    DEFAULT_CXX="${CXX:-$(command -v g++ || true)}"
    [[ -n "$DEFAULT_CXX" ]] || err "no C++ compiler found"
    DEFAULT_CXX_MAJOR="$($DEFAULT_CXX -dumpversion | cut -d. -f1)"

    if ((DEFAULT_CXX_MAJOR > SUPPORTED_GCC_MAX)); then
        COMPATIBLE_CXX=""
        for ((version = SUPPORTED_GCC_MAX; version >= 11; version--)); do
            if command -v "g++-$version" >/dev/null 2>&1; then
                COMPATIBLE_CXX="$(command -v "g++-$version")"
                break
            fi
        done

        if [[ -n "$COMPATIBLE_CXX" ]]; then
            COMPATIBLE_CC="$(command -v "gcc-${COMPATIBLE_CXX##*-}" || true)"
            export CC="${COMPATIBLE_CC:-$COMPATIBLE_CXX}"
            export CXX="$COMPATIBLE_CXX"
            ok "Using $COMPATIBLE_CXX as the CUDA host compiler"
        else
            warn "$DEFAULT_CXX is GCC $DEFAULT_CXX_MAJOR; CUDA $CUDA_MAJOR.x supports GCC through $SUPPORTED_GCC_MAX. The build will use --allow-unsupported-compiler and may still fail."
        fi
    fi
fi

if [[ -n "$ARCH_OVERRIDE" ]]; then
    export TORCH_CUDA_ARCH_LIST="$ARCH_OVERRIDE"
elif [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    DETECTED_ARCH="$("$VENV_PY" -c \
        'import torch; print(".".join(map(str, torch.cuda.get_device_capability()))) if torch.cuda.is_available() else None' \
        2>/dev/null || true)"
    [[ -n "$DETECTED_ARCH" ]] \
        || err "GPU architecture could not be detected; pass --arch (RTX 50-series uses --arch 12.0)"
    export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCH"
fi
ok "Building for CUDA architecture $TORCH_CUDA_ARCH_LIST"

UV_BIN="$(command -v uv || true)"
[[ -n "$UV_BIN" ]] || err "uv is required to install the custom rasterizer into the project venv"

info "Building and installing custom_rasterizer (CUDA)"
"$UV_BIN" pip install \
    --python "$VENV_PY" \
    --no-build-isolation \
    --no-deps \
    --reinstall \
    "$SCRIPT_DIR/hy3dgen/texgen/custom_rasterizer" \
    || err "custom_rasterizer build failed"

info "Verifying native-extension imports"
"$VENV_PY" - <<'PY'
import torch
import custom_rasterizer
import custom_rasterizer_kernel
from hy3dgen.texgen.differentiable_renderer import mesh_processor

for module in (custom_rasterizer, custom_rasterizer_kernel, mesh_processor):
    print(f"    imported {module.__name__}: {module.__file__}")
PY

ok "All native extensions built and imported"
