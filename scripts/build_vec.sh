#!/bin/bash

# Usage: ./build_env.sh pong [local|fast|web]

RAYLIB_NAME='raylib-5.5_linux_amd64'
LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"

FLAGS=(
    -shared
    -Wall
    -I./$RAYLIB_NAME/include
    "pufferlib/extensions/test_binding.c" -o "test_binding.so"
    $LINK_ARCHIVES
    -lGL
    -lm
    -lpthread
    -ferror-limit=3
    -DPLATFORM_DESKTOP
    # Bite me
    -Werror=incompatible-pointer-types
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
    -fsanitize=address,undefined,bounds,pointer-overflow,leak
    -fno-omit-frame-pointer
    -fPIC
)

clang -g -O0 ${FLAGS[@]}
