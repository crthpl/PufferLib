# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
bash build.sh ENV_NAME              # Build specific env (e.g., mcenv, cartpole)
bash build.sh ENV_NAME --float      # fp32 instead of default bf16
bash build.sh ENV_NAME --debug      # Debug build
bash build.sh ENV_NAME --cpu        # CPU-only (no CUDA)
```

Output: `pufferlib/_C.cpython-*.so`. Uses ccache for incremental builds.

## Train / Eval / Sweep

```bash
uv run puffer train ENV_NAME                          # Basic training
uv run puffer train mcenv --wandb --wandb-name NAME   # With W&B logging
uv run puffer train mcenv --train.total-timesteps 200000000
uv run puffer eval mcenv --load-model-path PATH       # Evaluate checkpoint
uv run puffer sweep mcenv --wandb                     # Hyperparameter sweep
```

Multi-GPU: `torchrun --nproc-per-node=N -m pufferlib.pufferl train ENV_NAME`

Config overrides use dot notation: `--train.learning-rate 0.001 --env.max-ticks 2000`

Always use `--wandb` with a descriptive `--wandb-name` for training runs. Use names that describe what's being tested (e.g., `ablate-no-block`, `speed-3.75-fall-125`, `sweep-repro-1e46cuax`). This keeps the wandb project navigable — hundreds of unnamed runs make the dashboard unusable.

## Architecture

PufferLib 4.0 is a GPU-native RL training framework. Three layers:

1. **Python frontend** (`pufferlib/pufferl.py`): CLI, config loading, W&B, checkpoints
2. **CUDA backend** (`src/pufferlib.cu`, `src/bindings.cu` → `_C.so`): PPO training loop, vectorized env stepping, loss computation
3. **Environments** (`ocean/ENV/`): Each env is a C header + binding that compiles into the `.so`

### Environment structure

Each environment in `ocean/ENV/` has:
- `binding.c`: PufferLib binding — defines `OBS_SIZE`, `NUM_ATNS`, `ACT_SIZES`, implements `my_init`, `my_log`, and optionally GPU-native hooks (`my_gpu_native_init/step/reset/close/log`)
- `ENV.h` or source files: Environment logic (step, reset, observations, rewards)
- `cuda/` (optional): GPU-native kernels for zero-copy stepping

GPU-native envs (`#define MY_GPU_NATIVE` in binding.c) run entirely on GPU — no CPU-GPU data copies during stepping. The CUDA kernel processes all agents in parallel.

### Config system

INI files in `config/ENV.ini` with sections: `[base]`, `[vec]`, `[env]`, `[policy]`, `[train]`, `[sweep]`.

- `[env]` keys become `Dict* kwargs` passed to `my_init()` and `my_gpu_native_init()`
- `[sweep.SECTION.PARAM]` defines hyperparameter search distributions

### Policy networks

`pufferlib/models.py`: Encoder (Linear) → Network (MinGRU/MLP/LSTM) → Decoder (Linear → action heads + value).

Action heads are multi-discrete: `ACT_SIZES` array defines the number of options per action dimension.

## mcenv (Minecraft bridging environment)

Primary env under active development. Agent starts on a 1-block platform over void, must bridge to navigation targets.

Key files:
- `ocean/mcenv/mcenv.h`: CPU env (eval/render path) — rewards, observations, phase logic, HUD
- `ocean/mcenv/binding.c`: PufferLib binding + GPU-native hooks
- `ocean/mcenv/cuda/mcenv_cuda.cu`: CUDA step kernel (training path) — must match mcenv.h exactly
- `ocean/mcenv/cuda/mcenv_cuda.cuh`: Shared structs (McEnvState, McEnvConfig, McEnvLog), constants, helpers
- `ocean/mcenv/cuda/mcenv_env.cuh`: CUDA reset + observation computation
- `config/mcenv.ini`: Reward weights, training hyperparams

Curriculum: phase 0 (block placement) → phase 1 (navigation + blocks) → phase 2 (navigation only). Phase transitions driven by `lifetime_blocks` and `avg_targets_ema` ratchets.

**When modifying rewards or observations**: changes must be made in BOTH `mcenv_cuda.cu` (GPU training) and `mcenv.h` (CPU eval/render) to stay in sync.

## Behavioral Cloning Pipeline

For warmstarting RL with demonstration data (e.g., sprint-jump-bridging demos):

### 1. Record demonstrations

Use the mcenv-codex demo_3d binary (`~/dev/mcenv-codex`):
```bash
cd ~/dev/mcenv-codex && cargo run --bin demo_3d --release
```
- Play the game, press **L** to start/stop recording, **Shift+L** to discard
- Recordings saved as `recordings/NNN_seconds:ticks.mce` (MCREC001 binary format)
- Format: initial state + blocks, then Input/Tick/Target events

### 2. Train BC policy

```bash
uv run python tools/bc_train.py recordings/*.mce --output bc_weights.bin --epochs 100
```
- Parses MCREC001 recordings, converts to obs/action sequences
- Trains encoder + MinGRU + decoder with cross-entropy loss
- Full recurrent training via MinGRU parallel scan (not feedforward)
- Overlapping windows (stride = seq_len/2) for data augmentation
- **Constants in bc_train.py must match current mcenv obs/action layout** (MC_OBS_TOTAL, ACT_SIZES, YAW_DELTAS, etc.)

### 3. RL fine-tune from BC weights

```bash
uv run puffer train mcenv --load-model-path bc_weights.bin --wandb
```

Weight format: flat float32, no biases, order: encoder weight → decoder weight → MinGRU layer weights.
