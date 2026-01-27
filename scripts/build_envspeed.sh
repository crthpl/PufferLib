#!/bin/bash

# Usage: ./build_envspeed.sh

RAYLIB_NAME='raylib-5.5_linux_amd64'
LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"

FLAGS=(
    -Wall
    -I./$RAYLIB_NAME/include
    -I/usr/local/cuda/include
    -Ipufferlib/extensions
    "pufferlib/extensions/test_envspeed.c"
    "pufferlib/extensions/ini.c"
    -o "test_envspeed"
    $LINK_ARCHIVES
    -lGL
    -lm
    -lpthread
    -ldl
    -L/usr/local/cuda/lib64 -lcudart
    -ferror-limit=3
    -DPLATFORM_DESKTOP
    # Bite me
    -Werror=incompatible-pointer-types
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
    -fms-extensions
    #-fsanitize=address,undefined,bounds,pointer-overflow,leak
    #-fno-omit-frame-pointer
    #-fsanitize=thread
    -fPIC
)

clang -O2 -DNDEBUG -fopenmp ${FLAGS[@]}
