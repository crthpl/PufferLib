#!/bin/bash
# Profile CUDA kernels with nsys and print grouped kernel summary
#
# Usage: ./profile_kernels.sh [timing|nsys|all]
#   timing - Run all profiles without nsys (fast, just timings)
#   nsys   - Run composite operations with nsys kernel breakdown
#   all    - Run both (default)

MODE=${1:-all}

run_timing() {
    local name=$1
    ./profile_kernels_torch "$name"
}

run_nsys() {
    local name=$1
    echo "=== $name (nsys kernel breakdown) ==="
    nsys profile \
        --capture-range=cudaProfilerApi \
        --cuda-graph-trace=node \
        --stats=false \
        --force-overwrite=true \
        -o nprof-kernels \
        ./profile_kernels_torch "$name" 2>&1 | grep -v "^Generating\|^Processing"

    nsys stats --report cuda_gpu_kern_sum:base --force-export=true nprof-kernels.nsys-rep 2>/dev/null | tail -n +4
    echo ""
}

if [[ "$MODE" == "timing" || "$MODE" == "all" ]]; then
    echo "========== TIMING ONLY =========="
    echo ""
    run_timing kernels
    run_timing forwardcall
    run_timing trainforward
    run_timing rolloutcopy
    run_timing envspeed
fi

if [[ "$MODE" == "nsys" || "$MODE" == "all" ]]; then
    echo "========== NSYS KERNEL BREAKDOWN =========="
    echo ""
    run_nsys forwardcall
    run_nsys trainforward
    run_nsys rolloutcopy
    run_nsys envspeed
fi
