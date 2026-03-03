## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import json
import ast
import time
import argparse
import configparser
from collections import defaultdict
import multiprocessing as mp
from copy import deepcopy

import numpy as np

import torch
import pufferlib
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import PufferLib C++ backend. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

def abbreviate(num, b2, c2):
    prefixes = ['', 'K', 'M', 'B', 'T']
    for i, prefix in enumerate(prefixes):
        if num < 1e3: break
        num /= 1e3

    return f'{b2}{num:.1f}{c2}{prefix}'

def duration(seconds, b2, c2):
    if seconds < 0: return f"{b2}0{c2}s"
    if seconds < 1: return f"{b2}{seconds*1000:.0f}{c2}ms"
    seconds = int(seconds)
    d = f'{b2}{seconds // 86400}{c2}d '
    h = f'{b2}{(seconds // 3600) % 24}{c2}h '
    m = f'{b2}{(seconds // 60) % 60}{c2}m '
    s = f'{b2}{seconds % 60}{c2}s'
    return d + h + m + s

def fmt_perf(name, color, delta_ref, elapsed, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*elapsed/delta_ref - 1e-5)
    return f'{color}{name}', duration(elapsed, b2, c2), f'{b2}{percent:2d}{c2}%'

def print_dashboard(args, model_size, logs, clear=False, idx=[0],
        c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
    console = rich.console.Console()
    dashboard = Table(box=rich.box.ROUNDED, expand=True,
        show_header=False, border_style='bright_cyan')
    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)

    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=18)
    table.add_column(justify="right", width=12)

    util = logs['utilization']
    table.add_row(
        f'{b1}PufferLib {b2}4.0 {idx[0]*" "}:blowfish:',
        f'{c1}GPU: {b2}{util.get("gpu_util", 0):.0f}{c2}%',
        f'{c1}VRAM: {b2}{util.get("vram_used_gb", 0):.1f}{c2}/{b2}{util.get("vram_total_gb", 0):.0f}{c2}G',
        f'{c1}RAM: {b2}{util.get("cpu_mem_gb", 0):.1f}{c2}G',
    )
    idx[0] = (idx[0] - 1) % 10

    s = Table(box=None, expand=True)
    remaining = f'{b2}A hair past a freckle{c2}'
    agent_steps = logs['agent_steps']
    if logs['SPS'] != 0:
        remaining = duration((args['train']['total_timesteps']*args['train']['gpus'] - agent_steps)/logs['SPS'], b2, c2)

    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
    s.add_row(f'{c2}Env', f'{b2}{args["env_name"]}')
    s.add_row(f'{c2}Params', abbreviate(model_size, b2, c2))
    s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
    s.add_row(f'{c2}SPS', abbreviate(logs['SPS'], b2, c2))
    s.add_row(f'{c2}Epoch', f'{b2}{logs['epoch']}')
    s.add_row(f'{c2}Uptime', duration(logs['uptime'], b2, c2))
    s.add_row(f'{c2}Remaining', remaining)

    perf = logs['performance']
    delta = perf.get('rollout', 0) + perf.get('train', 0)
    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf('Evaluate', b1, delta, perf.get('rollout', 0), b2, c2))
    p.add_row(*fmt_perf('  GPU', b2, delta, perf.get('eval_gpu', 0), b2, c2))
    p.add_row(*fmt_perf('  Env', b2, delta, perf.get('eval_env', 0), b2, c2))
    p.add_row(*fmt_perf('Train', b1, delta, perf.get('train', 0), b2, c2))
    p.add_row(*fmt_perf('  Misc', b2, delta, perf.get('train_misc', 0), b2, c2))
    p.add_row(*fmt_perf('  Forward', b2, delta, perf.get('train_forward', 0), b2, c2))

    l = Table(box=None, expand=True)
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for metric, value in logs['losses'].items():
        l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)

    for i, (metric, value) in enumerate(logs['environment'].items()):
        u = left if i % 2 == 0 else right
        u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
        if i == 30:
            break

    if clear:
        console.clear()

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())

def validate_config(args):
    minibatch_size = args['train']['minibatch_size']
    horizon = args['train']['horizon']
    total_agents = args['vec']['total_agents']
    if (minibatch_size % horizon) != 0:
        raise pufferlib.APIUsageError(
            f'minibatch_size {minibatch_size} must be divisible by horizon {horizon}')
    if minibatch_size > horizon * total_agents:
        raise pufferlib.APIUsageError(
            f'minibatch_size {minibatch_size} > total_agents {total_agents} * horizon {horizon}')

def downsample(ary, n):
    if not ary or n <= 0: return []
    if n == 1: return ary[-1:]
    if len(ary) <= n: return ary
    ary, last = np.array(ary[:-1]), ary[-1]
    n_trunc = (n-1) * (len(ary) // (n-1))
    ary = ary[-n_trunc:] if n_trunc > 0 else ary
    return ary.reshape(n-1, -1).mean(axis=1).tolist() + [last]

def _train(env_name, args, sweep_obj=None, result_queue=None, verbose=False):
    '''Single-GPU training worker. Process target for both DDP ranks and sweep trials.'''
    rank = args['rank']
    run_id = str(int(1000*time.time()))
    if rank == 0 and args['wandb']:
        import wandb
        run_id = wandb.util.generate_id()
        wandb.init(id=run_id, config=args,
            project=args['wandb_project'], group=args['wandb_group'],
            tags=[args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),
        )

    checkpoint_dir = os.path.join(args['data_dir'], args['env_name'], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    target_key = f'environment/{args["sweep"]["metric"]}'
    total_timesteps = args['train']['total_timesteps']
    all_logs = []

    pufferl = _C.create_pufferl(args)
    model_size = pufferl.num_params()

    if verbose:
        print_dashboard(args, model_size, _C.log(pufferl), clear=True)

    while pufferl.global_step < total_timesteps:
        _C.rollouts(pufferl)
        _C.train(pufferl)

        if rank != 0:
            continue

        # Rate-limit dashboard/logging to avoid overhead
        done_training = pufferl.global_step >= total_timesteps

        if pufferl.epoch % args['checkpoint_interval'] == 0 or done_training:
            model_path = os.path.join(checkpoint_dir, f'model_{args["env_name"]}_{pufferl.epoch:06d}.bin')
            _C.save_weights(pufferl, model_path)

        if done_training or time.time() > pufferl.last_log_time + 0.6:
            logs = _C.log(pufferl)
            flat_logs = dict(pufferlib.unroll_nested_dict(logs))

            if target_key in flat_logs and pufferl.global_step > min(0.20*total_timesteps, 100_000_000):
                all_logs.append(flat_logs)
                if sweep_obj is not None and sweep_obj.early_stop(flat_logs, target_key):
                    break

            if verbose:
                print_dashboard(args, model_size, logs)

            if args['wandb']:
                wandb.log(flat_logs, step=flat_logs['agent_steps'])

    if rank != 0:
        return _C.close(pufferl)

    for i in range(128): # Accurate eval for final log point
        _C.rollouts(pufferl)
        if i == 0 or i % 32 != 0: continue
        logs = _C.log(pufferl)
        if logs['environment']:
            break

    flat_logs = dict(pufferlib.unroll_nested_dict(logs))
    if target_key in flat_logs:
        all_logs.append(flat_logs)

    if args['wandb']:
        wandb.log(flat_logs, step=flat_logs['agent_steps'])

    if verbose:
        print_dashboard(args, model_size, logs)

    _C.close(pufferl)

    # Downsample results
    points_per_run = args['sweep']['downsample']
    scores = downsample([log[target_key] for log in all_logs], points_per_run)
    costs = downsample([log['uptime'] for log in all_logs], points_per_run)
    timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

    # Save own log: config + downsampled results
    log_dir = os.path.join(args['data_dir'], 'logs', args['env_name'])
    os.makedirs(log_dir, exist_ok=True)
    log_data = {
        **dict(pufferlib.unroll_nested_dict(args)),
        'uptime': flat_logs['uptime'],
        'scores': scores, 'costs': costs, 'timesteps': timesteps,
    }
    with open(os.path.join(log_dir, run_id + '.json'), 'w') as f:
        json.dump(log_data, f)

    if args['wandb']:
        if sweep_obj is None: # Don't spam uploads during sweeps
            artifact = wandb.Artifact(run_id, type='model')
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)

        wandb.run.finish()

    if result_queue is not None:
        result_queue.put((args['gpu_id'], scores, costs, timesteps))

def train(env_name, args=None, gpus=None, **kwargs):
    args = args or load_config(env_name)
    validate_config(args)

    subprocess = gpus is not None
    gpus = list(gpus or range(args['train']['gpus']))
    args['vec']['num_threads'] //= len(gpus)
    args['world_size'] = len(gpus)
    args['nccl_id_path'] = f'/tmp/puffer_nccl_{os.getpid()}_{gpus[0]}'

    if not subprocess:
        gpus = gpus[-1:] + gpus[:-1]  # Main process gets rank 0

    ctx = mp.get_context('spawn')
    for rank, gpu_id in reversed(list(enumerate(gpus))):
        worker_args = deepcopy(args)
        worker_args['rank'] = rank
        worker_args['gpu_id'] = gpu_id
        if rank == 0 and not subprocess:
            _train(env_name, worker_args, verbose=True)
        else:
            ctx.Process(target=_train, args=(env_name, worker_args), kwargs=kwargs).start()

def sweep(env_name, args=None, pareto=False):
    '''Train entry point. Handles single-GPU, multi-GPU DDP, and sweeps.'''
    args = args or load_config(env_name)
    exp_gpus = args['train']['gpus']
    sweep_gpus = args['sweep']['gpus'] or len(os.listdir('/proc/driver/nvidia/gpus'))
    args['vec']['num_threads'] //= (sweep_gpus // exp_gpus)
    args['no_model_upload'] = True

    sweep_config = args['sweep']
    method = sweep_config.pop('method')
    import pufferlib.sweep
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep_obj = sweep_cls(sweep_config)
    num_experiments = args['sweep']['max_runs']
    ts_config = sweep_config['train']['total_timesteps']
    all_timesteps = np.geomspace(ts_config['min'], ts_config['max'], sweep_gpus)
    result_queue = mp.get_context('spawn').Queue()

    active = {}
    completed = 0
    while completed < num_experiments:
        if len(active) >= sweep_gpus//exp_gpus: # Collect completed runs
            gpu_id, scores, costs, timesteps = result_queue.get()
            done_args = active.pop(gpu_id)

            if not scores:
                sweep_obj.observe(done_args, 0, 0, is_failure=True)
            else:
                completed += 1

            for s, c, t in zip(scores, costs, timesteps):
                done_args['train']['total_timesteps'] = t
                sweep_obj.observe(done_args, s, c, is_failure=False)

        idx = completed + len(active)
        if idx >= num_experiments:
            break # All experiments launched

        # TODO: only 1 per sweep etc
        gpu_id = next(i for i in range(sweep_gpus) if i not in active)
        timestep_total = all_timesteps[gpu_id] if pareto else None
        if idx > 1: # First experiment uses defaults
            sweep_obj.suggest(args, fixed_total_timesteps=timestep_total)

        try:
            validate_config(args)
        except pufferlib.APIUsageError as e:
            print(f'WARNING: {e}, skipping')
            sweep_obj.observe(args, 0, 0, is_failure=True)
            continue

        exp_args = deepcopy(args)
        active[gpu_id] = exp_args
        train(env_name, exp_args, range(gpu_id, gpu_id + exp_gpus),
            sweep_obj=sweep_obj, result_queue=result_queue)

def eval(env_name, args=None, load_path=None):
    '''Evaluate a trained policy using the native pipeline.
    Creates a full PuffeRL instance, optionally loads weights, then
    runs rollouts in a loop with rendering on env 0.'''
    args = args or load_config(env_name)

    pufferl_cpp = _C.create_pufferl(args)

    # Resolve load path
    load_path = load_path or args.get('load_model_path')
    if load_path == 'latest':
        data_dir = args['data_dir']
        pattern = os.path.join(data_dir, args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        if not candidates:
            raise FileNotFoundError(f'No .bin checkpoints found in {data_dir}/{args["env_name"]}/')
        load_path = max(candidates, key=os.path.getctime)

    if load_path is not None:
        _C.load_weights(pufferl_cpp, load_path)
        print(f'Loaded weights from {load_path}')

    while True:
        _C.render(pufferl_cpp, 0)
        _C.rollouts(pufferl_cpp)

    _C.close(pufferl_cpp)

def load_config(env_name):
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandbrun')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='puffer4')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    #CC: Remove the default. Just raise an error on "puffer train" etc with no env (think we already do)
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    for section in p.sections():
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except:
                value = p[section][key]

            #TODO: Can clean up with default sections in 3.13+
            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            dtype = type(value)
            parser.add_argument(
                fmt.replace('_', '-'), default=value,
                type=lambda v, t=dtype: v if v == 'auto' else t(v),
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['use_rnn'] = args['rnn_name'] is not None
    return dict(args)

def main():
    err = 'Usage: puffer [train, eval, sweep, paretosweep] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if 'train' in mode:
        train(env_name=env_name)
    elif 'eval' in mode:
        eval(env_name=env_name)
    elif 'sweep' in mode:
        sweep(env_name=env_name, pareto='pareto' in mode)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
