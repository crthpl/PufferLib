#!/usr/bin/env python3
"""Convert a native PufferLib CUDA checkpoint (.bin) to a PyTorch state dict.

The native backend saves a flat float32 array of weights only (no biases).
The PyTorch --slowly backend expects a state dict with both weights and biases.
This script loads the raw .bin, maps weights to the PyTorch policy's parameter
names (skipping biases), and zeros all biases, then saves a proper .pt file
that --slowly can load directly.

Usage:
    python convert_checkpoint.py <input.bin> [output.pt] [--env mcenv]

If output is omitted, writes to <input>.pt alongside the .bin file.
"""
import argparse
import sys
import numpy as np
import torch

import pufferlib
import pufferlib.pufferl
import pufferlib.models


def convert(bin_path, pt_path, env_name):
    args = pufferlib.pufferl.load_config(env_name)
    from pufferlib import _C

    obs_size = _C.get_obs_size() if hasattr(_C, 'get_obs_size') else None
    act_sizes = _C.get_act_sizes() if hasattr(_C, 'get_act_sizes') else None

    # If _C doesn't expose these (cpu build), infer from config
    if obs_size is None:
        # Fall back: create a vec just to read obs_size/act_sizes
        vec = _C.create_vec(args, 0)
        obs_size = vec.obs_size
        act_sizes = list(vec.act_sizes)
        vec.close()

    policy_cfg = args.get('policy', {})
    hidden_size = policy_cfg.get('hidden_size', 128)
    num_layers = policy_cfg.get('num_layers', 4)

    torch_cfg = args.get('torch', {})
    network_cls = getattr(pufferlib.models, torch_cfg.get('network', 'MinGRU'))
    encoder_cls = getattr(pufferlib.models, torch_cfg.get('encoder', 'DefaultEncoder'))
    decoder_cls = getattr(pufferlib.models, torch_cfg.get('decoder', 'DefaultDecoder'))

    encoder = encoder_cls(obs_size, hidden_size)
    decoder = decoder_cls(act_sizes, hidden_size)
    network = network_cls(hidden_size, num_layers)
    policy = pufferlib.models.Policy(encoder, decoder, network)

    # Load raw float32 weights
    raw = np.fromfile(bin_path, dtype=np.float32)
    flat = torch.from_numpy(raw)

    print(f'Checkpoint: {len(raw)} float32 values ({len(raw)*4} bytes)')

    # Map to state dict: weights only, biases zeroed
    offset = 0
    state_dict = {}
    weight_params = []
    bias_params = []

    for name, param in policy.named_parameters():
        if 'bias' in name:
            # Zero bias
            state_dict[name] = torch.zeros_like(param)
            bias_params.append(name)
            continue
        if name == 'decoder.decoder_logstd':
            # logstd is a parameter but not a weight matrix — read from checkpoint
            n = param.numel()
            state_dict[name] = flat[offset:offset+n].reshape(param.shape)
            offset += n
            weight_params.append(f'{name} [{n}]')
            continue
        n = param.numel()
        if offset + n > len(flat):
            print(f'ERROR: checkpoint too small! Need {offset+n} values, have {len(flat)}')
            sys.exit(1)
        state_dict[name] = flat[offset:offset+n].reshape(param.shape)
        offset += n
        weight_params.append(f'{name} [{n}]')

    if offset != len(flat):
        print(f'WARNING: {len(flat) - offset} unused values at end of checkpoint')

    policy.load_state_dict(state_dict)

    print(f'\nLoaded weights ({offset} values):')
    for w in weight_params:
        print(f'  {w}')
    print(f'\nZeroed biases:')
    for b in bias_params:
        print(f'  {b}')

    torch.save(policy.state_dict(), pt_path)
    print(f'\nSaved: {pt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Path to native .bin checkpoint')
    parser.add_argument('output', nargs='?', help='Output .pt path (default: <input>.pt)')
    parser.add_argument('--env', default='mcenv', help='Environment name for config lookup')
    args = parser.parse_args()

    # Prevent pufferl.load_config from parsing sys.argv
    sys.argv = [sys.argv[0]]

    pt_path = args.output or args.input.rsplit('.', 1)[0] + '.pt'
    convert(args.input, pt_path, args.env)
