// profiling/main.cu
// Main dispatcher for PufferLib CUDA profiling
//
// Build without torch:
//   nvcc -O3 -arch=sm_89 -DPRECISION_FLOAT -I. profiling/main.cu -o profile_v2
//
// Build with torch (use setup.py):
//   python setup.py build_profiler_v2 --env=breakout
//
// Usage:
//   ./profile_v2 <profile>
//   ./profile_v2 kernels          # All individual kernel profiles
//   ./profile_v2 trainforward     # Training forward+backward breakdown
//   ./profile_v2 trainstep        # Full training step with Muon optimizer
//   ./profile_v2 rolloutcopy      # Per-minibatch data prep: advantage+prio+copy
//   ./profile_v2 forwardcall      # Inference forward pass
//   ./profile_v2 envspeed         # Environment throughput
//   ./profile_v2 all              # Everything

// Include all profile modules (each is #pragma once guarded)
#include "profile_mingru.cu"
#include "profile_logcumsumexp.cu"
#include "profile_fusedscan.cu"
#include "profile_fcmax.cu"
#include "profile_ppoloss.cu"
#include "profile_sample.cu"
#include "profile_forward.cu"
#include "profile_envspeed.cu"
#include "profile_rolloutcopy.cu"
#include "profile_train.cu"

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels        - All individual kernel microbenchmarks\n");
    printf("  mingrugate     - MinGRU gate kernel only\n");
    printf("  logcumsumexp   - Logcumsumexp kernel only\n");
    printf("  fusedscan      - Fused scan (checkpointed) kernel only\n");
    printf("  samplelogits   - Sample logits kernel only\n");
    printf("  ppoloss        - PPO loss kernel only\n");
    printf("  fcmax          - FC+Max kernel only\n");
#ifdef USE_TORCH
    printf("  forwardcall    - Inference forward pass (requires torch)\n");
    printf("  trainforward   - Training fwd+loss+bwd breakdown (requires torch)\n");
    printf("  trainstep      - Full training step with Muon optimizer (requires torch)\n");
    printf("  rolloutcopy    - Per-minibatch data prep: advantage+prio+copy (requires torch)\n");
#endif
#ifdef USE_STATIC_ENV
    printf("  envspeed       - Environment step throughput (requires static env)\n");
#endif
    printf("  all            - Run all available profiles\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* profile = argv[1];
    warmup_gpu();

    // Using typical breakout settings: INPUT_SIZE=96, H=128, A=4
    bool run_all = strcmp(profile, "all") == 0;

    // === Individual kernel microbenchmarks ===
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "mingrugate") == 0 || run_all) {
        profile_mingrugate(BR, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "logcumsumexp") == 0 || run_all) {
        profile_logcumsumexp(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fusedscan") == 0 || run_all) {
        profile_fusedscan(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "samplelogits") == 0 || run_all) {
        profile_samplelogits(BR, A);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all) {
        profile_ppoloss(BT, T, A);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fcmax") == 0 || run_all) {
        profile_fcmax(BR, 63, 7, 128);    // partner encoder (drive)
        profile_fcmax(BR, 200, 13, 128);  // road encoder (drive)
    }

    // === Composite profiles (require torch) ===
#ifdef USE_TORCH
    if (strcmp(profile, "forwardcall") == 0 || run_all) {
        profile_forwardcall(BR, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainforward") == 0 || run_all) {
        profile_trainforward(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainstep") == 0 || run_all) {
        profile_trainstep(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "rolloutcopy") == 0 || run_all) {
        // num_segments = BR*BUF (full rollout), minibatch_segs = BT
        profile_rolloutcopy(BR * BUF, T, BT, INPUT_SIZE, A, 1, H);
    }
#endif

    // === Environment speed (requires static env link) ===
#ifdef USE_STATIC_ENV
    if (strcmp(profile, "envspeed") == 0 || run_all) {
        profile_envspeed(BUF*BR, BUF, 16, T);
    }
#endif

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "logcumsumexp") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "fcmax") != 0
#ifdef USE_TORCH
        && strcmp(profile, "forwardcall") != 0
        && strcmp(profile, "trainforward") != 0
        && strcmp(profile, "trainstep") != 0
        && strcmp(profile, "rolloutcopy") != 0
#endif
#ifdef USE_STATIC_ENV
        && strcmp(profile, "envspeed") != 0
#endif
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
