#!/bin/bash
set -e

# Usage:
#   ./scripts/build.sh breakout              # Build _C.so with breakout statically linked
#   ./scripts/build.sh breakout --float      # float32 precision (required for --slowly)
#   ./scripts/build.sh breakout --debug      # Debug build
#   ./scripts/build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./scripts/build.sh breakout --fast       # Standalone executable (optimized)
#   ./scripts/build.sh breakout --web        # Emscripten web build

ENV=${1:?Usage: ./scripts/build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web]}
MODE=""
PRECISION=""
DEBUG=""
for arg in "${@:2}"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
    esac
done

# ============================================================================
# Platform + dependencies
# ============================================================================

PLATFORM="$(uname -s)"
RAYLIB_NAME='raylib-5.5_macos'
BOX2D_NAME='box2d-macos-arm64'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    BOX2D_NAME='box2d-linux-amd64'
fi
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    BOX2D_NAME='box2d-web'
fi

RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
SRC_DIR="ocean/$ENV"

download() {
    local name=$1 ext=$2
    if [ ! -d "$name" ]; then
        echo "Downloading $name..."
        if [ "$ext" = ".zip" ]; then
            curl -sL "$3/$name$ext" -o "$name$ext" && unzip -q "$name$ext" && rm "$name$ext"
        else
            curl -sL "$3/$name$ext" -o "$name$ext" && tar xf "$name$ext" && rm "$name$ext"
        fi
    fi
}

download "$RAYLIB_NAME" ".tar.gz" "$RAYLIB_URL"
[ ! -f "$RAYLIB_NAME/include/rlights.h" ] && \
    curl -sL "https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h" \
        -o "$RAYLIB_NAME/include/rlights.h"
download "$BOX2D_NAME" ".tar.gz" "$BOX2D_URL"
[ "$MODE" = "web" ] && download "raylib-5.5_webassembly" ".zip" "$RAYLIB_URL"
[ "$MODE" = "web" ] && download "box2d-web" ".tar.gz" "$BOX2D_URL"

# Shared include paths
INCLUDES=(-I./$RAYLIB_NAME/include -I./$BOX2D_NAME/include -I./$BOX2D_NAME/src -I./puffernet)

# Box2d link archive (impulse_wars needs it)
LINK_ARCHIVES="$RAYLIB_A"
[ "$ENV" = "impulse_wars" ] && LINK_ARCHIVES="$LINK_ARCHIVES ./$BOX2D_NAME/libbox2d.a"

# Extra source files
EXTRA_SRC=""
[ "$ENV" = "constellation" ] && EXTRA_SRC="$SRC_DIR/cJSON.c"

# ============================================================================
# Standalone builds: --local, --fast, --web
# ============================================================================

if [ "$MODE" = "web" ]; then
    [ ! -f "minshell.html" ] && \
        curl -sL "https://raw.githubusercontent.com/raysan5/raylib/master/src/minshell.html" -o minshell.html
    mkdir -p "build_web/$ENV"
    echo "Building $ENV for web..."
    emcc \
        -o "build_web/$ENV/game.html" \
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
        -O3 -Wall \
        $LINK_ARCHIVES \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file ./minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared
    echo "Built: build_web/$ENV/game.html"
    exit 0
fi

if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        -Wall
        "${INCLUDES[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "$ENV"
        $LINK_ARCHIVES
        -lGL -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
        -ferror-limit=3
        -Werror=incompatible-pointer-types
        -Werror=return-type
        -Wno-error=incompatible-pointer-types-discards-qualifiers
        -Wno-incompatible-pointer-types-discards-qualifiers
        -Wno-error=array-parameter
    )
    if [ "$PLATFORM" = "Darwin" ]; then
        FLAGS+=(-framework Cocoa -framework IOKit -framework CoreVideo)
    fi
    if [ "$MODE" = "local" ]; then
        echo "Building $ENV (debug)..."
        [ "$PLATFORM" = "Linux" ] && FLAGS+=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
        clang -g -O0 "${FLAGS[@]}"
    else
        echo "Building $ENV (optimized)..."
        clang -O2 -DNDEBUG "${FLAGS[@]}"
    fi
    echo "Built: ./$ENV"
    exit 0
fi

# ============================================================================
# Default: build _C.so with env statically linked
# ============================================================================

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
NVCC="$CUDA_HOME/bin/nvcc"

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

if [ -n "$DEBUG" ]; then
    OPT="-O0 -g"; LINK_OPT="-g"
else
    OPT="-O3"; LINK_OPT="-O2"
fi

# Step 1: Build static env library
BINDING_SRC="ocean/$ENV/binding.c"
STATIC_OBJ="src/libstatic_${ENV}.o"
STATIC_LIB="src/libstatic_${ENV}.a"

[ ! -f "$BINDING_SRC" ] && echo "Error: $BINDING_SRC not found" && exit 1

echo "=== Building static env: $ENV ==="
clang -c -O2 -DNDEBUG \
    -I. -Isrc -Iocean/$ENV \
    -I./$RAYLIB_NAME/include -I$CUDA_HOME/include \
    -DPLATFORM_DESKTOP \
    -fno-semantic-interposition -fvisibility=hidden \
    -fPIC -fopenmp \
    "$BINDING_SRC" -o "$STATIC_OBJ"

ar rcs "$STATIC_LIB" "$STATIC_OBJ"

OBS_TENSOR_T=$(strings "$STATIC_OBJ" | grep 'Tensor$' | head -1)
[ -z "$OBS_TENSOR_T" ] && echo "Error: Could not find OBS_TENSOR_T in $STATIC_OBJ" && exit 1
echo "OBS_TENSOR_T=$OBS_TENSOR_T"

# Step 2: Compile bindings.cu
echo "=== Compiling bindings.cu ==="
NVCC_CMD=(
    $NVCC -c -Xcompiler -fPIC
    -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1
    -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    -Xcompiler=-DPLATFORM_DESKTOP
    -std=c++17
    -I. -Isrc
    -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE
    -I$CUDA_HOME/include -I$RAYLIB_NAME/include
    -Xcompiler=-fopenmp
    -DOBS_TENSOR_T=$OBS_TENSOR_T
)
[ -n "$PRECISION" ] && NVCC_CMD+=($PRECISION)
NVCC_CMD+=($OPT src/bindings.cu -o src/bindings.o)

echo "${NVCC_CMD[@]}"
"${NVCC_CMD[@]}"

# Step 3: Link
echo "=== Linking $OUTPUT ==="
LINK_CMD=(
    g++ -shared -fPIC -fopenmp
    src/bindings.o "$STATIC_LIB" "$RAYLIB_A"
    -L$CUDA_HOME/lib64
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
    -lnvToolsExt -lomp5
    $LINK_OPT
)
if [ "$PLATFORM" = "Linux" ]; then
    LINK_CMD+=(-Bsymbolic-functions)
elif [ "$PLATFORM" = "Darwin" ]; then
    LINK_CMD+=(-framework Cocoa -framework OpenGL -framework IOKit)
fi
LINK_CMD+=(-o "$OUTPUT")

echo "${LINK_CMD[@]}"
"${LINK_CMD[@]}"

echo "=== Built: $OUTPUT ==="
