#!/usr/bin/env python3
"""Behavioral cloning trainer for mcenv.

Reads mcenv-codex MCREC001 recordings, converts to obs/action pairs,
and trains the PufferLib policy with cross-entropy loss.
Saves weights in .bin format compatible with --load-model-path.

Usage:
    uv run python tools/bc_train.py recording1.bin [recording2.bin ...] \
        --output bc_weights.bin --epochs 100
"""
import argparse
import struct
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
                changes = []
                for _ in range(n):
                    changes.append({
                        'x': read_i32(f), 'y': read_i32(f), 'z': read_i32(f),
                        'solid': read_bool(f)
                    })
                events.append(('tick', state, changes))
            else:
                raise ValueError(f"Unknown tag: {tag:#x}")
    return initial_player, initial_blocks, events

# ---------------------------------------------------------------------------
# Convert recording to obs/action pairs
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


def input_to_actions(inp, prev_yaw, prev_pitch):
    actions = np.zeros(NUM_ATNS, dtype=np.int64)
    actions[0] = int(round(inp['forward'])) + 1
    actions[1] = int(round(inp['strafe'])) + 1
    actions[2] = 1 if inp['jump'] else 0
    actions[3] = 1 if inp['sneak'] else 0
    actions[4] = 1 if inp['sprint'] else 0

    if inp['has_look']:
        yaw_delta = inp['look_yaw'] - prev_yaw
        pitch_delta = inp['look_pitch'] - prev_pitch
    else:
        yaw_delta = 0.0
        pitch_delta = 0.0

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


def recording_to_dataset(path, target_x=None, target_z=None):
    initial_player, initial_blocks, events = load_recording(path)
    blocks = set(initial_blocks)
    yaw = initial_player['yaw']
    pitch = initial_player['pitch']
    if target_x is None: target_x = 40.5
    if target_z is None: target_z = 40.5

    all_obs, all_actions = [], []
    current_state = initial_player

    for event in events:
        if event[0] == 'input':
            inp = event[1]
            obs = compute_obs(current_state, yaw, pitch, target_x, target_z, blocks)
            actions = input_to_actions(inp, yaw, pitch)
            all_obs.append(obs)
            all_actions.append(actions)
            rot_pct = actions[8] * 0.25
            yaw += YAW_DELTAS[actions[5]] * rot_pct
            pitch += PITCH_DELTAS[actions[6]] * rot_pct
            pitch = max(-90.0, min(90.0, pitch))
        elif event[0] == 'tick':
            state, changes = event[1], event[2]
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

# ---------------------------------------------------------------------------
# Policy model — exact architecture match for PufferLib CUDA backend
#
# Weight layout in .bin: encoder | decoder | mingru_layer_0 | ... | mingru_layer_N
# - Encoder: (hidden_size, obs_size) — no bias
# - Decoder: (total_actions+1, hidden_size) — no bias, +1 for value
# - MinGRU: (3*hidden_size, hidden_size) per layer — no bias
#
# For BC we do feedforward: encoder -> relu -> mingru_layers_as_linear -> decoder
# The MinGRU layers are (3H, H) matrices. In feedforward mode we just use
# the first H rows as a linear transform (the "hidden" portion).
# This gives us weight-compatible initialization that RL can fine-tune.
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    def __init__(self, obs_size, act_sizes, hidden_size=256, num_layers=4):
        super().__init__()
        self.act_sizes = list(act_sizes)
        self.hidden_size = hidden_size
        total_actions = sum(act_sizes)

        # Encoder: (hidden_size, obs_size) — matches CUDA
        self.encoder_weight = nn.Parameter(torch.empty(hidden_size, obs_size))

        # Decoder: (total_actions+1, hidden_size) — matches CUDA
        self.decoder_weight = nn.Parameter(torch.empty(total_actions + 1, hidden_size))

        # MinGRU layers: (3*hidden_size, hidden_size) each — matches CUDA
        self.gru_weights = nn.ParameterList([
            nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, obs):
        # Encoder
        h = torch.relu(obs @ self.encoder_weight.t())

        # MinGRU layers used as feedforward (just the first H rows = "hidden" transform)
        for w in self.gru_weights:
            H = self.hidden_size
            # w is (3H, H). Split into hidden(H,H), gate(H,H), proj(H,H)
            w_hidden = w[:H]     # (H, H)
            w_proj = w[2*H:3*H]  # (H, H) — highway projection
            hidden_out = torch.relu(h @ w_hidden.t())
            proj = torch.sigmoid(h @ w_proj.t())
            h = proj * hidden_out + (1.0 - proj) * h  # highway connection

        # Decoder
        out = h @ self.decoder_weight.t()  # (B, total_actions+1)
        action_logits = out[:, :-1]  # drop value head
        return list(action_logits.split(self.act_sizes, dim=1))

    def save_bin(self, path):
        """Save weights in flat .bin format matching PufferLib CUDA layout."""
        flat = []
        # Order: encoder, decoder, gru layers
        flat.append(self.encoder_weight.detach().cpu().numpy().flatten())
        flat.append(self.decoder_weight.detach().cpu().numpy().flatten())
        for w in self.gru_weights:
            flat.append(w.detach().cpu().numpy().flatten())
        flat = np.concatenate(flat).astype(np.float32)
        flat.tofile(path)
        print(f"Saved {path} ({flat.shape[0]} floats, {flat.nbytes} bytes)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(obs_data, act_data, output_path, hidden_size=256, num_layers=4,
             epochs=200, batch_size=512, lr=1e-3, device='cuda'):
    policy = BCPolicy(MC_OBS_TOTAL, ACT_SIZES, hidden_size, num_layers).to(device)

    obs_t = torch.from_numpy(obs_data).to(device)
    act_t = torch.from_numpy(act_data).to(device)
    dataset = torch.utils.data.TensorDataset(obs_t, act_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(obs_data)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Training BC: {n} samples, {n_params:,} params, {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = [0] * NUM_ATNS
        total_count = 0

        for batch_obs, batch_act in loader:
            logit_heads = policy(batch_obs)
            loss = sum(criterion(logits, batch_act[:, i])
                       for i, logits in enumerate(logit_heads))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.shape[0]
            for i, logits in enumerate(logit_heads):
                total_correct[i] += (logits.argmax(1) == batch_act[:, i]).sum().item()
            total_count += batch_obs.shape[0]

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
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    all_obs, all_acts = [], []
    for rec_path in args.recordings:
        print(f"Loading {rec_path}...")
        obs, acts = recording_to_dataset(rec_path, args.target_x, args.target_z)
        if len(obs) > 0:
            all_obs.append(obs)
            all_acts.append(acts)
            print(f"  {len(obs)} samples")

    if not all_obs:
        print("No samples!"); sys.exit(1)

    obs_data = np.concatenate(all_obs)
    act_data = np.concatenate(all_acts)
    print(f"Total: {len(obs_data)} samples")

    train_bc(obs_data, act_data, args.output,
             hidden_size=args.hidden_size, num_layers=args.num_layers,
             epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, device=args.device)


if __name__ == '__main__':
    main()
