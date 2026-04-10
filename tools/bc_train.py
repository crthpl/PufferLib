#!/usr/bin/env python3
"""Behavioral cloning trainer for mcenv.

Reads mcenv-codex MCREC001 recordings, converts to obs/action sequences,
and trains the PufferLib policy (encoder + MinGRU + decoder) with full
recurrent forward passes and cross-entropy loss.

Saves weights in .bin format compatible with --load-model-path.

Usage:
    uv run python tools/bc_train.py recording1.bin [recording2.bin ...] \
        --output bc_weights.bin --epochs 100
"""
import argparse
import struct
import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add parent to path so we can import pufferlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants (must match mcenv.h / binding.c)
# ---------------------------------------------------------------------------
MC_START_X = 25.5
MC_START_Y = 3.0
MC_START_Z = 25.5
MC_OBS_PLAYER = 14
MC_OBS_TARGET = 2
MC_GRID_X = 7
MC_GRID_Y = 3
MC_GRID_Z = 7
MC_OBS_GRID = MC_GRID_X * MC_GRID_Y * MC_GRID_Z
MC_OBS_TOTAL = MC_OBS_PLAYER + MC_OBS_TARGET + MC_OBS_GRID

ACT_SIZES = [3, 3, 2, 2, 2, 7, 7, 3, 5]
NUM_ATNS = len(ACT_SIZES)

YAW_DELTAS = [-180.0, -15.0, -1.0, 0.0, 1.0, 15.0, 180.0]
PITCH_DELTAS = [-180.0, -15.0, -1.0, 0.0, 1.0, 15.0, 180.0]

# ---------------------------------------------------------------------------
# Recording parser (mirrors recording.rs binary format)
# ---------------------------------------------------------------------------

def read_f64(f): return struct.unpack('<d', f.read(8))[0]
def read_f32(f): return struct.unpack('<f', f.read(4))[0]
def read_i32(f): return struct.unpack('<i', f.read(4))[0]
def read_u32(f): return struct.unpack('<I', f.read(4))[0]
def read_u8(f):  return struct.unpack('<B', f.read(1))[0]
def read_bool(f): return read_u8(f) != 0

def read_player_state(f):
    return {
        'pos_x': read_f64(f), 'pos_y': read_f64(f), 'pos_z': read_f64(f),
        'prev_x': read_f64(f), 'prev_y': read_f64(f), 'prev_z': read_f64(f),
        'vel_x': read_f64(f), 'vel_y': read_f64(f), 'vel_z': read_f64(f),
        'yaw': read_f32(f), 'pitch': read_f32(f), 'on_ground': read_bool(f),
        'bb_min_x': read_f64(f), 'bb_min_y': read_f64(f), 'bb_min_z': read_f64(f),
        'bb_max_x': read_f64(f), 'bb_max_y': read_f64(f), 'bb_max_z': read_f64(f),
        'sneaking': read_bool(f), 'sprinting': read_bool(f),
        'sprint_toggle_timer': read_u8(f), 'prev_forward_positive': read_bool(f),
    }

def read_input(f):
    return {
        'forward': read_f32(f), 'strafe': read_f32(f),
        'jump': read_bool(f), 'sprint': read_bool(f), 'sneak': read_bool(f),
        'has_look': read_bool(f), 'look_yaw': read_f32(f), 'look_pitch': read_f32(f),
        'place': read_bool(f), 'place_repeat': read_bool(f), 'break_block': read_bool(f),
    }

def load_recording(path):
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == b'MCREC001', f"Bad magic: {magic}"
        initial_player = read_player_state(f)
        num_blocks = read_u32(f)
        initial_blocks = set()
        for _ in range(num_blocks):
            initial_blocks.add((read_i32(f), read_i32(f), read_i32(f)))
        events = []
        while True:
            tag_bytes = f.read(1)
            if not tag_bytes:
                break
            tag = tag_bytes[0]
            if tag == 0x00:
                events.append(('input', read_input(f)))
            elif tag == 0x01:
                state = read_player_state(f)
                n = read_u32(f)
                changes = [{'x': read_i32(f), 'y': read_i32(f), 'z': read_i32(f),
                            'solid': read_bool(f)} for _ in range(n)]
                events.append(('tick', state, changes))
            elif tag == 0x02:
                tx, tz = read_f32(f), read_f32(f)
                events.append(('target', tx, tz))
            else:
                raise ValueError(f"Unknown tag: {tag:#x}")
    return initial_player, initial_blocks, events

# ---------------------------------------------------------------------------
# Convert recording to obs/action sequences
# ---------------------------------------------------------------------------

def compute_obs(state, yaw, pitch, target_x, target_z, blocks):
    obs = np.zeros(MC_OBS_TOTAL, dtype=np.float32)
    idx = 0
    obs[idx] = (state['pos_x'] - MC_START_X) / 50.0; idx += 1
    obs[idx] = (state['pos_y'] - MC_START_Y) / 10.0; idx += 1
    obs[idx] = (state['pos_z'] - MC_START_Z) / 10.0; idx += 1
    obs[idx] = state['pos_x'] - math.floor(state['pos_x']); idx += 1
    obs[idx] = state['pos_y'] - math.floor(state['pos_y']); idx += 1
    obs[idx] = state['pos_z'] - math.floor(state['pos_z']); idx += 1
    obs[idx] = state['vel_x']; idx += 1
    obs[idx] = state['vel_y']; idx += 1
    obs[idx] = state['vel_z']; idx += 1
    yaw_rad = yaw * math.pi / 180.0
    pitch_rad = pitch * math.pi / 180.0
    obs[idx] = math.sin(yaw_rad); idx += 1
    obs[idx] = math.cos(yaw_rad); idx += 1
    obs[idx] = math.sin(pitch_rad); idx += 1
    obs[idx] = math.cos(pitch_rad); idx += 1
    obs[idx] = 1.0 if state['on_ground'] else 0.0; idx += 1
    obs[idx] = (target_x - state['pos_x']) / 50.0; idx += 1
    obs[idx] = (target_z - state['pos_z']) / 50.0; idx += 1
    bx = int(math.floor(state['pos_x']))
    by = int(math.floor(state['pos_y']))
    bz = int(math.floor(state['pos_z']))
    for dx in range(-(MC_GRID_X // 2), MC_GRID_X // 2 + 1):
        for dy in range(-MC_GRID_Y, 0):
            for dz in range(-(MC_GRID_Z // 2), MC_GRID_Z // 2 + 1):
                obs[idx] = 1.0 if (bx+dx, by+dy, bz+dz) in blocks else 0.0
                idx += 1
    return obs


def recording_to_sequence(path, target_x=None, target_z=None):
    """Convert a recording to a single (obs_seq, act_seq) sequence.

    Handles fps != tps correctly:
    - fps > tps: multiple INPUTs before TICK — uses last input, but computes
      yaw/pitch delta from tick states (captures accumulated camera movement)
    - fps < tps: multiple TICKs per INPUT — emits zero-rotation action for
      extra ticks, reusing movement/button state from the input
    """
    initial_player, initial_blocks, events = load_recording(path)
    blocks = set(initial_blocks)

    # Track yaw/pitch from tick states for accurate delta computation
    prev_tick_yaw = initial_player['yaw']
    prev_tick_pitch = initial_player['pitch']

    # Default target; overridden by Target events in recording
    if target_x is None: target_x = 40.5
    if target_z is None: target_z = 40.5

    all_obs, all_actions = [], []
    current_state = initial_player
    pending_input = None

    for event in events:
        if event[0] == 'input':
            pending_input = event[1]
        elif event[0] == 'target':
            target_x = event[1]
            target_z = event[2]
        elif event[0] == 'tick':
            state, changes = event[1], event[2]

            # Compute yaw/pitch delta from tick states (handles multi-input accumulation)
            tick_yaw = state['yaw']
            tick_pitch = state['pitch']

            if pending_input is not None:
                # Compute action using tick-state yaw/pitch delta
                yaw_delta = tick_yaw - prev_tick_yaw
                pitch_delta = tick_pitch - prev_tick_pitch
                obs = compute_obs(current_state, prev_tick_yaw, prev_tick_pitch,
                                  target_x, target_z, blocks)
                actions = input_to_actions_with_delta(
                    pending_input, yaw_delta, pitch_delta)
                all_obs.append(obs)
                all_actions.append(actions)
                pending_input = None
            else:
                # Extra tick without new input (fps < tps): zero rotation,
                # reuse last movement/button state
                obs = compute_obs(current_state, prev_tick_yaw, prev_tick_pitch,
                                  target_x, target_z, blocks)
                # No-op rotation action: yaw=3(none), pitch=3(none), rot=0
                actions = np.zeros(NUM_ATNS, dtype=np.int64)
                actions[0] = 1  # forward=none
                actions[1] = 1  # strafe=none
                actions[5] = 3  # yaw=none
                actions[6] = 3  # pitch=none
                all_obs.append(obs)
                all_actions.append(actions)

            prev_tick_yaw = tick_yaw
            prev_tick_pitch = tick_pitch

            for c in changes:
                key = (c['x'], c['y'], c['z'])
                if c['solid']:
                    blocks.add(key)
                else:
                    blocks.discard(key)
            current_state = state

    if not all_obs:
        return np.array([]), np.array([])
    return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.int64)


def input_to_actions_with_delta(inp, yaw_delta, pitch_delta):
    """Convert PlayerInput to discrete actions, using pre-computed yaw/pitch deltas."""
    actions = np.zeros(NUM_ATNS, dtype=np.int64)
    actions[0] = int(round(inp['forward'])) + 1
    actions[1] = int(round(inp['strafe'])) + 1
    actions[2] = 1 if inp['jump'] else 0
    actions[3] = 1 if inp['sneak'] else 0
    actions[4] = 1 if inp['sprint'] else 0

    best_yaw_idx, best_pitch_idx, best_rot_idx = 3, 3, 4
    best_error = abs(yaw_delta) + abs(pitch_delta)
    for rot_idx in range(5):
        rot_pct = rot_idx * 0.25
        if rot_pct == 0.0:
            err = abs(yaw_delta) + abs(pitch_delta)
            if err < best_error:
                best_error = err
                best_yaw_idx, best_pitch_idx, best_rot_idx = 3, 3, 0
            continue
        for yi, yd in enumerate(YAW_DELTAS):
            for pi, pd in enumerate(PITCH_DELTAS):
                err = abs(yaw_delta - yd * rot_pct) + abs(pitch_delta - pd * rot_pct)
                if err < best_error:
                    best_error = err
                    best_yaw_idx, best_pitch_idx, best_rot_idx = yi, pi, rot_idx

    actions[5] = best_yaw_idx
    actions[6] = best_pitch_idx
    actions[7] = 2 if inp['place_repeat'] else (1 if inp['place'] else 0)
    actions[8] = best_rot_idx
    return actions

# ---------------------------------------------------------------------------
# Policy — uses PufferLib's exact MinGRU implementation
# ---------------------------------------------------------------------------

from pufferlib.models import MinGRU

class BCPolicy(nn.Module):
    """Exact architecture match for PufferLib CUDA backend.

    Weight layout in .bin: encoder | decoder | mingru_layer_0 | ... | mingru_layer_N
    All weights are (out, in) with no bias.
    """
    def __init__(self, obs_size, act_sizes, hidden_size=256, num_layers=4):
        super().__init__()
        self.act_sizes = list(act_sizes)
        self.hidden_size = hidden_size
        total_actions = sum(act_sizes)

        self.encoder = nn.Linear(obs_size, hidden_size, bias=False)
        self.network = MinGRU(hidden_size, num_layers=num_layers)
        # Decoder: fused (total_actions + 1) for logits + value
        self.decoder = nn.Linear(hidden_size, total_actions + 1, bias=False)

    def forward(self, obs_seq):
        """Forward pass on a sequence.

        Args:
            obs_seq: (B, T, obs_size) tensor

        Returns:
            list of (B*T, act_size_i) logit tensors per action head
        """
        B, T, _ = obs_seq.shape
        h = self.encoder(obs_seq.reshape(B * T, -1))  # (B*T, H)
        h = self.network.forward_train(h.reshape(B, T, -1))  # (B, T, H)
        out = self.decoder(h.reshape(B * T, -1))  # (B*T, total_actions+1)
        action_logits = out[:, :-1]  # drop value head
        return list(action_logits.split(self.act_sizes, dim=1))

    def save_bin(self, path):
        """Save weights in flat .bin format matching PufferLib CUDA layout.

        Order: encoder weight, decoder weight, then each MinGRU layer weight.
        """
        flat = []
        flat.append(self.encoder.weight.detach().cpu().numpy().flatten())
        flat.append(self.decoder.weight.detach().cpu().numpy().flatten())
        for layer in self.network.layers:
            flat.append(layer.weight.detach().cpu().numpy().flatten())
        flat = np.concatenate(flat).astype(np.float32)
        flat.tofile(path)
        print(f"Saved {path} ({flat.shape[0]} floats, {flat.nbytes} bytes)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_batches(sequences, seq_len, batch_size, stride=None):
    """Chop sequences into overlapping fixed-length chunks and batch them."""
    if stride is None:
        stride = seq_len // 2  # 50% overlap by default
    chunks_obs, chunks_act = [], []
    for obs_seq, act_seq in sequences:
        T = len(obs_seq)
        for start in range(0, T - seq_len + 1, stride):
            chunks_obs.append(obs_seq[start:start + seq_len])
            chunks_act.append(act_seq[start:start + seq_len])
    if not chunks_obs:
        return [], []
    obs = np.stack(chunks_obs)  # (N, T, obs_size)
    act = np.stack(chunks_act)  # (N, T, num_atns)
    # Shuffle
    perm = np.random.permutation(len(obs))
    obs, act = obs[perm], act[perm]
    # Split into batches
    batches = []
    for i in range(0, len(obs), batch_size):
        batches.append((obs[i:i+batch_size], act[i:i+batch_size]))
    return batches


def train_bc(sequences, output_path, hidden_size=256, num_layers=4,
             epochs=200, batch_size=32, seq_len=128, lr=1e-3, device='cuda'):
    policy = BCPolicy(MC_OBS_TOTAL, ACT_SIZES, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_ticks = sum(len(s[0]) for s in sequences)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Training BC: {total_ticks} ticks across {len(sequences)} sequences")
    print(f"Policy: {n_params:,} params, seq_len={seq_len}, batch_size={batch_size}")

    for epoch in range(epochs):
        batches = make_batches(sequences, seq_len, batch_size)
        total_loss = 0.0
        total_correct = [0] * NUM_ATNS
        total_count = 0

        for obs_np, act_np in batches:
            obs_t = torch.from_numpy(obs_np).to(device)  # (B, T, obs)
            act_t = torch.from_numpy(act_np).to(device)  # (B, T, atns)
            B, T, _ = obs_t.shape

            logit_heads = policy(obs_t)  # list of (B*T, act_size_i)
            act_flat = act_t.reshape(B * T, -1)

            loss = sum(criterion(logits, act_flat[:, i])
                       for i, logits in enumerate(logit_heads))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B * T
            for i, logits in enumerate(logit_heads):
                total_correct[i] += (logits.argmax(1) == act_flat[:, i]).sum().item()
            total_count += B * T

        if total_count == 0:
            continue

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / total_count
            accs = [f"{c/total_count:.2f}" for c in total_correct]
            print(f"  epoch {epoch+1:4d}  loss={avg_loss:.4f}  acc=[{' '.join(accs)}]")

    policy.save_bin(output_path)
    return policy


def main():
    parser = argparse.ArgumentParser(description='Behavioral cloning for mcenv')
    parser.add_argument('recordings', nargs='+', help='MCREC001 recording files')
    parser.add_argument('--output', '-o', default='bc_weights.bin')
    parser.add_argument('--target-x', type=float, default=None)
    parser.add_argument('--target-z', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for recurrent training (matches RL horizon)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    sequences = []
    for rec_path in args.recordings:
        print(f"Loading {rec_path}...")
        obs, acts = recording_to_sequence(rec_path, args.target_x, args.target_z)
        if len(obs) > 0:
            sequences.append((obs, acts))
            print(f"  {len(obs)} ticks")

    if not sequences:
        print("No data!"); sys.exit(1)

    train_bc(sequences, args.output,
             hidden_size=args.hidden_size, num_layers=args.num_layers,
             epochs=args.epochs, batch_size=args.batch_size,
             seq_len=args.seq_len, lr=args.lr, device=args.device)


if __name__ == '__main__':
    main()
