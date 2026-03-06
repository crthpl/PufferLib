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
    #'g2048',
    #'moba',
    #'pong',
    #'tower_climb',
    #'grid',
    #'nmmo3',
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
    for fpath in glob.glob(path):
        if 'cache.json' in fpath:
            continue

        with open(fpath, 'r') as f:
            exp = json.load(f)

        sweep_metadata = exp.pop('sweep')

        n = 0
        for k, v in exp.pop('metrics').items():
            n = len(v)
            if k not in data:
                data[k] = []

            data[k].append(v)

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
    data = {}
    for name in env_names:
        env_data = cached_sweep_load(f'logs/puffer_{name}', name)
        data[name] = env_data
        sweep_metadata = env_data.pop('sweep')

        normed_env_data = []
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

            normed = (dat - mmin) / (mmax - mmin)
            normed_env_data.append(normed)

        normed = np.stack(normed_env_data, axis=1)

    from sklearn.manifold import TSNE
    proj = TSNE(n_components=2)
    reduced = None
    try:
        reduced = proj.fit_transform(normed)
    except ValueError:
        print('Warning: TSNE failed. Skipping TSNE')
        sz = len(normed)

    row = 0
    for env in env_names:
        #for i, hyper in enumerate(HYPERS):
        #    sz = len(data[env][hyper])
        #    data[env][hyper] = normed[row:row+sz, i].tolist()

        #sz = len(data[env]['agent_steps'])

        data[env] = {k: v for k, v in data[env].items() if k in ALL_KEYS}
        if reduced is not None:
            data[env]['tsne1'] = reduced[row:row+sz, 0].tolist()
            data[env]['tsne2'] = reduced[row:row+sz, 1].tolist()
        else:
            data[env]['tsne1'] = np.random.rand(sz).tolist()
            data[env]['tsne2'] = np.random.rand(sz).tolist()

        row += sz
        print(f'Env {env} has {sz} points')

    json.dump(data, open('all_cache.json', 'w'))

if __name__ == '__main__':
    compute_tsne()
