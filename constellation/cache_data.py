import numpy as np

import json
import glob
import os

env_names = sorted([
    'breakout',
    #'impulse_wars',
    #'pacman',
    #'tetris',
    #'g2048',
    #'moba',
    'pong',
    #'tower_climb',
    #'grid',
    'connect4',
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
    'train/horizon',
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
    'tsne1',
    'tsne2',
]

ALL_KEYS = HYPERS + METRICS

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v


def pareto_idx(steps, costs, scores):
    idxs = []
    for i in range(len(steps)):
        better = [scores[j] >= scores[i] and
            costs[j] < costs[i] and steps[j] < steps[i]
            for j in range(len(scores))]
        if not any(better):
            idxs.append(i)

    return idxs

def cached_load(path, env_name, cache):
    data = {}
    num_metrics = 0
    metric_keys = []
    for fpath in glob.glob(path):
        if fpath in cache:
            exp = cache[fpath]
        else:
            with open(fpath, 'r') as f:
                try:
                    exp = json.load(f)
                except json.decoder.JSONDecodeError:
                    print(f'Skipping {fpath}')
                    continue

        cache[fpath] = exp

        if 'metrics' not in exp:
            print(f'Skipping {fpath} (no metrics)')
            continue

        # Temporary: Some experiments are missing loss keys
        for k in list(exp['metrics'].keys()):
            if 'loss' in k:
                del exp['metrics'][k]

        if num_metrics == 0:
            num_metrics = len(exp['metrics'])
            metric_keys = list(exp['metrics'].keys())

        skip = False
        metrics = exp['metrics']

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
                pass

        sweep_metadata = exp['sweep']

        for k, v in unroll_nested_dict(exp):
            if k not in data:
                data[k] = []

            data[k].append([v]*n)

        for hyper in HYPERS:
            prefix, suffix = hyper.split('/')
            #if prefix not in sweep_metadata:
            #    continue

            group = sweep_metadata[prefix]
            #if suffix not in group:
            #    continue


            key = f'{prefix}/{suffix}_norm'
            if key not in data:
                data[key] = []

            if suffix in group:
                param = group[suffix]
                mmin = param['min']
                mmax = param['max']
                dist = param['distribution']
                val = exp[prefix][suffix]

                if 'log' in dist or 'pow2' in dist:
                    mmin = np.log(mmin)
                    mmax = np.log(mmax)
                    val = np.log(val)

                norm = (val - mmin) / (mmax - mmin)
                data[key].append([norm]*n)
            else:
                data[key].append([1]*n)

    for k, v in data.items():
        data[k] = [item for sublist in v for item in sublist]

    for k in list(data.keys()):
        if 'sweep' in k:
            del data[k]

    # Format im millions to avoid overfloat in C
    data['agent_steps'] = [e/1e6 for e in data['agent_steps']]
    data['train/total_timesteps'] = [e/1e6 for e in data['train/total_timesteps']]
    #data['metrics/agent_steps'] = [e/1e6 for e in data['metrics/agent_steps']]
    del data['metrics/agent_steps']

    # Filter to pareto
    '''
    steps = data['agent_steps']
    costs = data['uptime']
    scores = data['env/score']
    idxs = pareto_idx(steps, costs, scores)
    for k in data:
        try:
            data[k] = [data[k][i] for i in idxs]
        except IndexError:
            breakpoint()
    '''

    data['sweep'] = sweep_metadata
    return data

def compute_tsne():
    all_data = {}
    normed = {}

    cache = {}
    cache_file = os.path.join('cache.json')
    if os.path.exists(cache_file):
        cache = json.load(open(cache_file, 'r'))

    for env in env_names:
        all_data[env] = cached_load(f'logs/puffer_{env}/*.json', env, cache)

    with open(cache_file, 'w') as f:
        json.dump(cache, f)

    for env in env_names:
        env_data = all_data[env]
        normed_env = []
        for key in HYPERS:
            norm_key = f'{key}_norm'
            normed_env.append(np.array(env_data[norm_key]))

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
        all_data[env]['tsne1'] = reduced[row:row+sz, 0].tolist()
        all_data[env]['tsne2'] = reduced[row:row+sz, 1].tolist()

        '''
        if reduced is not None:
            all_data[env]['tsne1'] = reduced[row:row+sz, 0].tolist()
            all_data[env]['tsne2'] = reduced[row:row+sz, 1].tolist()
        else:
            all_data[env]['tsne1'] = np.random.rand(sz).tolist()
            all_data[env]['tsne2'] = np.random.rand(sz).tolist()
        '''

        row += sz
        print(f'Env {env} has {sz} points')

    for env in all_data:
        dat = all_data[env]
        dat = {k: v for k, v in dat.items() if isinstance(v, list)
                and len(v) > 0 and isinstance(v[0], (int, float))
                and (k == 'train/max_grad_norm' or not k.endswith('_norm'))}
        all_data[env] = dat
        for k, v in dat.items():
            try:
                print(f'{env}/{k}: {len(v), min(v), max(v)}')
            except:
                print(f'{env}/{k}: {len(v)}')

    for env in all_data:
        for k, v in all_data[env].items():
            if isinstance(v, list):
                try:
                    all_data[env][k] = ','.join([f'{x:.6g}' for x in v])
                except:
                    breakpoint()

    json.dump(all_data, open('resources/constellation/experiments.json', 'w'))

if __name__ == '__main__':
    compute_tsne()
