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

ACT_SIZES = [3, 3, 2, 2, 2, 153, 107, 3]
NUM_ATNS = len(ACT_SIZES)

# Full angle tables: 0-16@1°, 16-90@2°, 90-180@4° (yaw only for coarse)
_fine = list(range(1, 17))
_medium = list(range(18, 91, 2))
_coarse = list(range(94, 179, 4)) + [180]
_yaw_pos = _fine + _medium + _coarse
YAW_DELTAS = [-v for v in reversed(_yaw_pos)] + [0.0] + [float(v) for v in _yaw_pos]
_pitch_pos = _fine + _medium
PITCH_DELTAS = [-v for v in reversed(_pitch_pos)] + [0.0] + [float(v) for v in _pitch_pos]
assert len(YAW_DELTAS) == 153
assert len(PITCH_DELTAS) == 107

# Index of 0.0 in each table (for no-rotation default)
YAW_ZERO = YAW_DELTAS.index(0.0)
PITCH_ZERO = PITCH_DELTAS.index(0.0)

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


def nearest_bucket(delta, table):
    """Find index of nearest value in a sorted table."""
    best_i, best_err = 0, abs(delta - table[0])
    for i in range(1, len(table)):
        err = abs(delta - table[i])
        if err < best_err:
            best_i, best_err = i, err
    return best_i


def recording_to_sequence(path, target_x=None, target_z=None):
    """Convert a recording to (obs_seq, act_seq, yaw_deltas, pitch_deltas).

    Returns continuous yaw/pitch deltas alongside discrete actions for soft-target training.
    """
    initial_player, initial_blocks, events = load_recording(path)
    blocks = set(initial_blocks)

    prev_tick_yaw = initial_player['yaw']
    prev_tick_pitch = initial_player['pitch']

    if target_x is None: target_x = 40.5
    if target_z is None: target_z = 40.5

    all_obs, all_actions, all_yaw_d, all_pitch_d = [], [], [], []
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
            tick_yaw = state['yaw']
            tick_pitch = state['pitch']

            if pending_input is not None:
                yaw_delta = tick_yaw - prev_tick_yaw
                pitch_delta = tick_pitch - prev_tick_pitch
                obs = compute_obs(current_state, prev_tick_yaw, prev_tick_pitch,
                                  target_x, target_z, blocks)
                actions = input_to_actions(pending_input, yaw_delta, pitch_delta)
                all_obs.append(obs)
                all_actions.append(actions)
                all_yaw_d.append(yaw_delta)
                all_pitch_d.append(pitch_delta)
                pending_input = None
            else:
                obs = compute_obs(current_state, prev_tick_yaw, prev_tick_pitch,
                                  target_x, target_z, blocks)
                actions = np.zeros(NUM_ATNS, dtype=np.int64)
                actions[0] = 1  # forward=none
                actions[1] = 1  # strafe=none
                actions[5] = YAW_ZERO
                actions[6] = PITCH_ZERO
                all_obs.append(obs)
                all_actions.append(actions)
                all_yaw_d.append(0.0)
                all_pitch_d.append(0.0)

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
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.array(all_obs, dtype=np.float32),
            np.array(all_actions, dtype=np.int64),
            np.array(all_yaw_d, dtype=np.float32),
            np.array(all_pitch_d, dtype=np.float32))


def input_to_actions(inp, yaw_delta, pitch_delta):
    """Convert PlayerInput to discrete actions with nearest-bucket angle lookup."""
    actions = np.zeros(NUM_ATNS, dtype=np.int64)
    actions[0] = int(round(inp['forward'])) + 1
    actions[1] = int(round(inp['strafe'])) + 1
    actions[2] = 1 if inp['jump'] else 0
    actions[3] = 1 if inp['sneak'] else 0
    actions[4] = 1 if inp['sprint'] else 0
    actions[5] = nearest_bucket(yaw_delta, YAW_DELTAS)
    actions[6] = nearest_bucket(pitch_delta, PITCH_DELTAS)
    actions[7] = 2 if inp['place_repeat'] else (1 if inp['place'] else 0)
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
    """Chop sequences into overlapping fixed-length chunks and batch them.

    Each sequence is (obs, act, yaw_d, pitch_d).
    """
    if stride is None:
        stride = seq_len // 2
    chunks_obs, chunks_act, chunks_yd, chunks_pd = [], [], [], []
    for obs_seq, act_seq, yd_seq, pd_seq in sequences:
        T = len(obs_seq)
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            chunks_obs.append(obs_seq[start:end])
            chunks_act.append(act_seq[start:end])
            chunks_yd.append(yd_seq[start:end])
            chunks_pd.append(pd_seq[start:end])
    if not chunks_obs:
        return []
    obs = np.stack(chunks_obs)
    act = np.stack(chunks_act)
    yd = np.stack(chunks_yd)
    pd = np.stack(chunks_pd)
    perm = np.random.permutation(len(obs))
    obs, act, yd, pd = obs[perm], act[perm], yd[perm], pd[perm]
    batches = []
    for i in range(0, len(obs), batch_size):
        batches.append((obs[i:i+batch_size], act[i:i+batch_size],
                        yd[i:i+batch_size], pd[i:i+batch_size]))
    return batches


def make_soft_targets(continuous_deltas, table_tensor, temperature=2.0):
    """Create soft targets from continuous deltas via inverse-distance weighting.

    Args:
        continuous_deltas: (N,) tensor of true continuous deltas
        table_tensor: (K,) tensor of bucket values
        temperature: controls sharpness (lower = sharper, 1.0 = linear interp)

    Returns:
        (N, K) soft target distribution
    """
    # (N, K) distance from each delta to each bucket
    dist = (continuous_deltas.unsqueeze(1) - table_tensor.unsqueeze(0)).abs()
    # Inverse distance weights with temperature
    weights = 1.0 / (dist / temperature + 1.0)
    return weights / weights.sum(dim=1, keepdim=True)


# Pre-compute table tensors for soft targets
YAW_TABLE = torch.tensor(YAW_DELTAS, dtype=torch.float32)
PITCH_TABLE = torch.tensor(PITCH_DELTAS, dtype=torch.float32)
# Indices of the angle heads in ACT_SIZES
YAW_HEAD = 5
PITCH_HEAD = 6


def eval_loss(policy, sequences, seq_len, batch_size, criterion, device):
    """Compute loss and accuracy on a set of sequences without gradient updates."""
    batches = make_batches(sequences, seq_len, batch_size, stride=seq_len)
    total_loss = 0.0
    total_correct = [0] * NUM_ATNS
    total_count = 0
    with torch.no_grad():
        for obs_np, act_np, _, _ in batches:
            obs_t = torch.from_numpy(obs_np).to(device)
            act_t = torch.from_numpy(act_np).to(device)
            B, T, _ = obs_t.shape
            logit_heads = policy(obs_t)
            act_flat = act_t.reshape(B * T, -1)
            loss = sum(criterion(logits, act_flat[:, i])
                       for i, logits in enumerate(logit_heads))
            total_loss += loss.item() * B * T
            for i, logits in enumerate(logit_heads):
                total_correct[i] += (logits.argmax(1) == act_flat[:, i]).sum().item()
            total_count += B * T
    if total_count == 0:
        return 0.0, [0.0] * NUM_ATNS
    return total_loss / total_count, [c / total_count for c in total_correct]


def train_bc(train_sequences, val_sequences, output_path, hidden_size=256, num_layers=4,
             epochs=200, batch_size=32, seq_len=128, stride=None, lr=1e-3,
             weight_decay=0.0, label_smoothing=0.0, soft_temp=2.0, device='cuda'):
    policy = BCPolicy(MC_OBS_TOTAL, ACT_SIZES, hidden_size, num_layers).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    yaw_table_d = YAW_TABLE.to(device)
    pitch_table_d = PITCH_TABLE.to(device)

    if stride is None:
        stride = seq_len // 2
    train_ticks = sum(len(s[0]) for s in train_sequences)
    val_ticks = sum(len(s[0]) for s in val_sequences) if val_sequences else 0
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Training BC: {train_ticks} train ticks, {val_ticks} val ticks "
          f"({len(train_sequences)} train / {len(val_sequences)} val sequences)")
    print(f"Policy: {n_params:,} params, seq_len={seq_len}, stride={stride}, batch_size={batch_size}")
    print(f"Soft targets: temp={soft_temp}, label_smoothing={label_smoothing}, weight_decay={weight_decay}")

    # Best checkpoint tracking
    best_val_loss = float('inf')
    best_epoch = -1

    # Epoch 0: evaluate before any training
    if val_sequences:
        val_loss, val_accs = eval_loss(policy, val_sequences, seq_len, batch_size, val_criterion, device)
        val_acc_str = ' '.join(f"{a:.2f}" for a in val_accs)
        print(f"  epoch    0  (init)  | val loss={val_loss:.4f}  acc=[{val_acc_str}]")
        best_val_loss = val_loss
        policy.save_bin(output_path)
        best_epoch = 0

    for epoch in range(epochs):
        batches = make_batches(train_sequences, seq_len, batch_size, stride=stride)
        total_loss = 0.0
        total_correct = [0] * NUM_ATNS
        total_count = 0

        for obs_np, act_np, yd_np, pd_np in batches:
            obs_t = torch.from_numpy(obs_np).to(device)
            act_t = torch.from_numpy(act_np).to(device)
            yd_t = torch.from_numpy(yd_np).to(device)
            pd_t = torch.from_numpy(pd_np).to(device)
            B, T, _ = obs_t.shape

            logit_heads = policy(obs_t)
            act_flat = act_t.reshape(B * T, -1)
            yd_flat = yd_t.reshape(B * T)
            pd_flat = pd_t.reshape(B * T)

            # Standard cross-entropy for non-angle heads
            loss = sum(criterion(logits, act_flat[:, i])
                       for i, logits in enumerate(logit_heads)
                       if i != YAW_HEAD and i != PITCH_HEAD)

            # Soft-target KL divergence for angle heads
            yaw_soft = make_soft_targets(yd_flat, yaw_table_d, soft_temp)
            pitch_soft = make_soft_targets(pd_flat, pitch_table_d, soft_temp)
            yaw_log_probs = F.log_softmax(logit_heads[YAW_HEAD], dim=1)
            pitch_log_probs = F.log_softmax(logit_heads[PITCH_HEAD], dim=1)
            loss += F.kl_div(yaw_log_probs, yaw_soft, reduction='batchmean')
            loss += F.kl_div(pitch_log_probs, pitch_soft, reduction='batchmean')

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
            line = f"  epoch {epoch+1:4d}  train loss={avg_loss:.4f}  acc=[{' '.join(accs)}]"
            if val_sequences:
                val_loss, val_accs = eval_loss(policy, val_sequences, seq_len, batch_size, val_criterion, device)
                val_acc_str = ' '.join(f"{a:.2f}" for a in val_accs)
                line += f"  | val loss={val_loss:.4f}  acc=[{val_acc_str}]"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    policy.save_bin(output_path)
                    line += "  *best*"
            print(line)

    if best_epoch >= 0:
        print(f"Best checkpoint: epoch {best_epoch}, val loss={best_val_loss:.4f}")
    else:
        policy.save_bin(output_path)
    return policy


def main():
    parser = argparse.ArgumentParser(description='Behavioral cloning for mcenv')
    parser.add_argument('recordings', nargs='+', help='MCREC001 recording files')
    parser.add_argument('--val', nargs='+', default=None,
                        help='Recordings to hold out for validation')
    parser.add_argument('--output', '-o', default='bc_weights.bin')
    parser.add_argument('--target-x', type=float, default=None)
    parser.add_argument('--target-z', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for recurrent training (matches RL horizon)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Window stride (default: seq_len/2, use --stride SEQ_LEN for no overlap)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--soft-temp', type=float, default=2.0,
                        help='Soft target temperature for angle heads (lower=sharper)')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    val_paths = set(os.path.abspath(p) for p in (args.val or []))

    train_sequences, val_sequences = [], []
    for rec_path in args.recordings:
        print(f"Loading {rec_path}...")
        obs, acts, yd, pd = recording_to_sequence(rec_path, args.target_x, args.target_z)
        if len(obs) > 0:
            if os.path.abspath(rec_path) in val_paths:
                val_sequences.append((obs, acts, yd, pd))
                print(f"  {len(obs)} ticks [val]")
            else:
                train_sequences.append((obs, acts, yd, pd))
                print(f"  {len(obs)} ticks [train]")

    if not train_sequences:
        print("No training data!"); sys.exit(1)

    train_bc(train_sequences, val_sequences, args.output,
             hidden_size=args.hidden_size, num_layers=args.num_layers,
             epochs=args.epochs, batch_size=args.batch_size,
             seq_len=args.seq_len, stride=args.stride,
             lr=args.lr, weight_decay=args.weight_decay,
             label_smoothing=args.label_smoothing, soft_temp=args.soft_temp,
             device=args.device)


if __name__ == '__main__':
    main()
