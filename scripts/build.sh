#!/bin/bash
set -e

# Usage:
#   ./scripts/build.sh                    # Build _C.so (no env linked)
#   ./scripts/build.sh breakout           # Build _C.so with breakout statically linked
#   ./scripts/build.sh breakout --float   # Build with float32 precision (required for --slowly)
#   ./scripts/build.sh breakout --debug   # Debug build

ENV=${1:-}
PRECISION=""
DEBUG=""
for arg in "$@"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
    esac
done

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
NVCC="$CUDA_HOME/bin/nvcc"

# Platform
PLATFORM="$(uname -s)"
RAYLIB_NAME='raylib-5.5_macos'
BOX2D_NAME='box2d-macos-arm64'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    BOX2D_NAME='box2d-linux-amd64'
fi
RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"

# Download raylib/box2d if missing
RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"

if [ ! -d "$RAYLIB_NAME" ]; then
    echo "Downloading $RAYLIB_NAME..."
    curl -L "$RAYLIB_URL/$RAYLIB_NAME.tar.gz" -o "$RAYLIB_NAME.tar.gz"
    tar xf "$RAYLIB_NAME.tar.gz" && rm "$RAYLIB_NAME.tar.gz"
    curl -L "https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h" \
        -o "$RAYLIB_NAME/include/rlights.h"
fi
if [ ! -d "raylib-5.5_webassembly" ]; then
    echo "Downloading raylib webassembly..."
    curl -L "$RAYLIB_URL/raylib-5.5_webassembly.zip" -o raylib-5.5_webassembly.zip
    unzip -q raylib-5.5_webassembly.zip && rm raylib-5.5_webassembly.zip
fi
if [ ! -d "$BOX2D_NAME" ]; then
    echo "Downloading $BOX2D_NAME..."
    curl -L "$BOX2D_URL/$BOX2D_NAME.tar.gz" -o "$BOX2D_NAME.tar.gz"
    tar xf "$BOX2D_NAME.tar.gz" && rm "$BOX2D_NAME.tar.gz"
fi
if [ ! -d "box2d-web" ]; then
    echo "Downloading box2d-web..."
    curl -L "$BOX2D_URL/box2d-web.tar.gz" -o box2d-web.tar.gz
    tar xf box2d-web.tar.gz && rm box2d-web.tar.gz
fi

# Python paths (the only things we need Python for)
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

# Compile flags
if [ -n "$DEBUG" ]; then
    OPT="-O0 -g"
    LINK_OPT="-g"
else
    OPT="-O3"
    LINK_OPT="-O2"
fi

STATIC_LIB=""
OBS_TENSOR_T=""

# Step 1: Build static env library (if env specified)
if [ -n "$ENV" ] && [ "$ENV" != "--float" ] && [ "$ENV" != "--debug" ]; then
    BINDING_SRC="ocean/$ENV/binding.c"
    STATIC_OBJ="src/libstatic_${ENV}.o"
    STATIC_LIB="src/libstatic_${ENV}.a"

    if [ ! -f "$BINDING_SRC" ]; then
        echo "Error: $BINDING_SRC not found"
        exit 1
    fi

    echo "=== Building static env: $ENV ==="
    clang -c -O2 -DNDEBUG \
        -I. -Isrc -Iocean/$ENV \
        -I./$RAYLIB_NAME/include -I$CUDA_HOME/include \
        -DPLATFORM_DESKTOP \
        -fno-semantic-interposition -fvisibility=hidden \
        -fPIC -fopenmp \
        "$BINDING_SRC" -o "$STATIC_OBJ"

    ar rcs "$STATIC_LIB" "$STATIC_OBJ"

    # Extract OBS_TENSOR_T from compiled object
    OBS_TENSOR_T=$(strings "$STATIC_OBJ" | grep 'Tensor$' | head -1)
    if [ -z "$OBS_TENSOR_T" ]; then
        echo "Error: Could not find OBS_TENSOR_T in $STATIC_OBJ"
        exit 1
    fi
    echo "OBS_TENSOR_T=$OBS_TENSOR_T"
fi

# Step 2: Compile bindings.cu → bindings.o
echo "=== Compiling bindings.cu ==="
NVCC_CMD=(
    $NVCC -c -Xcompiler -fPIC
    -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1
    -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    -Xcompiler=-DPLATFORM_DESKTOP
    -std=c++17
    -I. -Isrc
    -I$PYTHON_INCLUDE
    -I$PYBIND_INCLUDE
    -I$NUMPY_INCLUDE
    -I$CUDA_HOME/include
    -I$RAYLIB_NAME/include
    -Xcompiler=-fopenmp
)
[ -n "$PRECISION" ] && NVCC_CMD+=($PRECISION)
[ -n "$OBS_TENSOR_T" ] && NVCC_CMD+=("-DOBS_TENSOR_T=$OBS_TENSOR_T")
NVCC_CMD+=($OPT src/bindings.cu -o src/bindings.o)

echo "${NVCC_CMD[@]}"
"${NVCC_CMD[@]}"

# Step 3: Link → _C.so
echo "=== Linking $OUTPUT ==="
LINK_CMD=(
    g++ -shared -fPIC -fopenmp
    src/bindings.o
)
[ -n "$STATIC_LIB" ] && LINK_CMD+=($STATIC_LIB)
LINK_CMD+=(
    $RAYLIB_A
    -L$CUDA_HOME/lib64
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
    -lnvToolsExt -lomp5
    $LINK_OPT
)

# Platform-specific link flags
if [ "$PLATFORM" = "Linux" ]; then
    LINK_CMD+=(-Bsymbolic-functions)
elif [ "$PLATFORM" = "Darwin" ]; then
    LINK_CMD+=(-framework Cocoa -framework OpenGL -framework IOKit)
fi

LINK_CMD+=(-o "$OUTPUT")

echo "${LINK_CMD[@]}"
"${LINK_CMD[@]}"

echo "=== Built: $OUTPUT ==="
