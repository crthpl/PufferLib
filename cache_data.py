import numpy as np

import json
import glob
import os

import pufferlib


env_names = sorted([
    'breakout',
    #'impulse_wars',
    #'pacman',
    #'tetris',
    'g2048',
    #'moba',
    'pong',
    #'tower_climb',
    'grid',
    'nmmo3',
    #'snake',
    #'tripletriad'
])

HYPERS = [
    'train/learning_rate',
    'train/ent_coef',
    'train/gamma',
    'train/gae_lambda',
    'train/vtrace_rho_clip',
    'train/vtrace_c_clip',
    'train/clip_coef',
    'train/vf_clip_coef',
    'train/vf_coef',
    'train/max_grad_norm',
    'train/beta1',
    'train/beta2',
    'train/eps',
    'train/prio_alpha',
    'train/prio_beta0',
    #'train/horizon',
    'train/replay_ratio',
    'train/minibatch_size',
    'policy/hidden_size',
    'vec/total_agents',
]

METRICS = [
    'agent_steps',
    'uptime',
    'env/score',
    'env/perf',
]

ALL_KEYS = HYPERS + METRICS

def pareto_idx(steps, costs, scores):
    idxs = []
    for i in range(len(steps)):
        better = [scores[j] >= scores[i] and
            costs[j] < costs[i] and steps[j] < steps[i]
            for j in range(len(scores))]
        if not any(better):
            idxs.append(i)

    return idxs

def load_sweep_data(path):
    data = {}
    sweep_metadata = {}
    num_metrics = 0
    for fpath in glob.glob(path):
        if 'cache.json' in fpath:
            continue

        with open(fpath, 'r') as f:
            try:
                exp = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'Skipping {fpath}')
                continue

        sweep_metadata = exp.pop('sweep')

        data_len = len(exp['metrics']['agent_steps'])
        if data_len > 100:
            print(f'Skipping {fpath} (len={data_len})')
            continue

        if num_metrics == 0:
            num_metrics = len(exp['metrics'])

        skip = False
        metrics = exp.pop('metrics')

        if len(metrics) != num_metrics:
            print(f'Skipping {fpath} (num_metrics={len(metrics)} != {num_metrics})')
            continue

        n = len(metrics['agent_steps'])
        for k, v in metrics.items():
            if len(v) != n:
                skip = True
                break

            if k not in data:
                data[k] = []

            if np.isnan(v).any():
                skip = True
                break

        if skip:
            print(f'Skipping {fpath} (bad data)')
            continue

        for k, v in metrics.items():
            data[k].append(v)
            if len(data[k]) != len(data['SPS']):
                breakpoint()
                pass

        for k, v in pufferlib.unroll_nested_dict(exp):
            if k not in data:
                data[k] = []

            data[k].append([v]*n)

    for k, v in data.items():
        data[k] = [item for sublist in v for item in sublist]

    #steps = data['agent_steps']
    #costs = data['uptime']
    #scores = data['env/score']
    #idxs = pareto_idx(steps, costs, scores)
    # Filter to pareto
    #for k in data:
    #    data[k] = [data[k][i] for i in idxs]

    data['sweep'] = sweep_metadata
    return data

def cached_sweep_load(path, env_name):
    cache_file = os.path.join(path, 'c_cache.json')
    if not os.path.exists(cache_file):
        data = load_sweep_data(os.path.join(path, '*.json'))
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    with open(cache_file, 'r') as f:
        data = json.load(f)

    print(f'Loaded {env_name}')
    return data

def compute_tsne():
    all_data = {}
    normed = {}

    for env in env_names:
        env_data = cached_sweep_load(f'logs/puffer_{env}', env)
        sweep_metadata = env_data.pop('sweep')
        all_data[env] = env_data

        normed_env = []
        for key in HYPERS:
            prefix, suffix = key.split('/')
            mmin = sweep_metadata[prefix][suffix]['min']
            mmax = sweep_metadata[prefix][suffix]['max']
            dat = np.array(env_data[key])

            dist = sweep_metadata[prefix][suffix]['distribution']
            if 'log' in dist or 'pow2' in dist:
                mmin = np.log(mmin)
                mmax = np.log(mmax)
                dat = np.log(dat)

            normed_env.append((dat - mmin) / (mmax - mmin))

        normed[env] = np.stack(normed_env, axis=1)

    normed = np.concatenate(list(normed.values()), axis=0)

    from sklearn.manifold import TSNE
    proj = TSNE(n_components=2)
    reduced = None
    try:
        reduced = proj.fit_transform(normed)
    except ValueError:
        print('Warning: TSNE failed. Skipping TSNE')

    row = 0
    for env in env_names:
        sz = len(all_data[env]['agent_steps'])
        #all_data[env] = {k: v for k, v in all_data[env].items()}
        if reduced is not None:
            all_data[env]['tsne1'] = reduced[row:row+sz, 0].tolist()
            all_data[env]['tsne2'] = reduced[row:row+sz, 1].tolist()
        else:
            all_data[env]['tsne1'] = np.random.rand(sz).tolist()
            all_data[env]['tsne2'] = np.random.rand(sz).tolist()

        row += sz
        print(f'Env {env} has {sz} points')

    json.dump(all_data, open('all_cache.json', 'w'))

if __name__ == '__main__':
    compute_tsne()
