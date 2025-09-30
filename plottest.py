from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import json
import glob
import os

# Global styling variables
FONT_FAMILY = 'Arial'
FONT_SIZE_TITLE = 28
FONT_SIZE_AXIS = 22
FONT_SIZE_TICK = 20
FONT_SIZE_LEGEND = 18
FONT_COLOR = '#f1f1f1'
PLOT_BG_COLOR = '#061a1a'
PAPER_BG_COLOR = '#061a1a'
LINE_WIDTH = 4
LINE_COLORS = ["#0000b3", "#0010d9", "#0020ff", "#0040ff", "#0060ff", "#0080ff", "#009fff", "#00bfff", "#00ffff"][::-1]
#['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

TITLE_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TITLE,
    color=FONT_COLOR
)

AXIS_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_AXIS,
    color=FONT_COLOR
)

TICK_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TICK,
    color=FONT_COLOR
)

LEGEND_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_LEGEND,
    color=FONT_COLOR
)

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
    'train/adam_beta1',
    'train/adam_beta2',
    'train/adam_eps',
    'train/prio_alpha',
    'train/prio_beta0',
    'train/bptt_horizon',
    'train/num_minibatches',
    'train/minibatch_size',
    'policy/hidden_size',
    'env/frameskip',
    'env/num_envs',
]

ALL_KEYS = [
    'agent_steps',
    'cost',
    'environment/score'
] + HYPERS

def rgba(hex, alpha):
    return f"rgba({int(hex[1:3], 16)}, {int(hex[3:5], 16)}, {int(hex[5:7], 16)}, {alpha})"

def band(experiments, key, qmin=0.0, qmax=1.0):
    mmax = np.array(experiments[key]).max()
    top = qmax * mmax
    bot = qmin * mmax
    filtered = {k: [] for k in experiments}
    for i, score in enumerate(experiments[key]):
        if score < bot or score > top:
            continue

        for k, v in experiments.items():
            filtered[k].append(v[i])

    return filtered

def mean_conf(xx, yy):
    x_min = min([min(x) for x in xx])
    x_max = max([max(x) for x in xx])
    y_min = min([min(y) for y in yy])
    y_max = max([max(y) for y in yy])

    x = np.linspace(x_min, x_max, 100)
    y_interps = np.stack([
        np.interp(x, x_, y_) for x_, y_ in zip(xx, yy)])

    mean = np.mean(y_interps, axis=0)
    std = np.std(y_interps, axis=0)
    conf = 1.96 * std / np.sqrt(len(xx))

    return x, mean, conf

def figure(title='The Puffer Frontier Project',
           xlabel='Uptime', ylabel='Score',
           legend='Trial', xaxis_type='linear'):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        xaxis=dict(title=dict(text=xlabel, font=AXIS_FONT), tickfont=TICK_FONT),
        yaxis=dict(title=dict(text=ylabel, font=AXIS_FONT), tickfont=TICK_FONT),
        xaxis_type=xaxis_type,
        showlegend=True,
        legend=dict(font=LEGEND_FONT),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        width=1280,
        height=720,
        autosize=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def plot_lines(fig, xx, yy, name='Trial'):
    for i, (x, y) in enumerate(zip(xx, yy)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=name,
                line=dict(
                    color=LINE_COLORS[i % len(LINE_COLORS)],
                    width=LINE_WIDTH
                )
            )
         )

def scatter(fig, x, y, c, legend='Trial', log_x=False, i=0, showlegend=True):
    mmin = min(c)
    mmax = max(c)
    #vals = [(c - mmin)/(mmax - mmin) for c in c]
    #vals = [max(0.01, v) for v in vals]

    if isinstance(c, str):
        colors = c
    else:
        colors = []
        for e in c:
            if mmin != mmax:
                v = (e - mmin)/(mmax - mmin)
            else:
                v = e

            if v < 0.001:
                v = 0.001
            colors.append(f'rgb(0, 0.5, {v})')

    #c = (np.array(c) - min(c))/(max(c) - min(c))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=legend,
            showlegend=showlegend,
            marker=dict(
                color=colors,
                size=10
            )
        )
    )

def plot_group(fig, xx, yy, xlabel='Performance', legend='Trial', log_x=False, i=0):
    x, mean, conf = mean_conf(xx, yy)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean + conf, (mean - conf)[::-1]]),
            fill='toself',
            fillcolor=rgba(LINE_COLORS[i], 0.2),
            line=dict(
                color='rgba(255,255,255,0)',
                width=LINE_WIDTH
            ),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode='lines',
            name=legend,
            line=dict(
                color=LINE_COLORS[i],
                width=LINE_WIDTH
            )
        )
    )

def plot_quantiles(fig, experiments, thresh, xlabel='Performance', legend='Trial', log_x=False, i=0):
    # Define quantile thresholds (in descending order for proper range filtering)
    #quantile_thresholds = [1.0, 0.95, 0.9, 0.75, 0.5, 0.25, 0.05]
    quantile_thresholds = [1.0, thresh]
    colors = [['indigo', 'blue', 'green', 'yellow', 'orange', 'red'][i]]

    for i in range(len(quantile_thresholds) - 1):
        qmin = quantile_thresholds[i + 1]
        qmax = quantile_thresholds[i]

        filtered = band(experiments, 'environment/score', qmin, qmax)

        if len(filtered['environment/score']) < 5:
            continue  # Skip

        steps  = filtered['agent_steps']
        # Adjust steps
        if 'env/frameskip' in experiments:
            skip = experiments['env/frameskip']
            steps = [n*m for n, m in zip(steps, skip)]

        x = steps
        y = filtered['cost']
        s = np.ones_like(x)

        fx, fy, fs = pareto_points(x, y, s, 0.05)

        fx = np.array(fx)
        fy = np.array(fy)
        fs = np.array(fs)

        px, py, y_lower, y_upper = loess_fit(fx, fs, fy, n_bins=10, frac=0.4)

        #x_q = x[mask]
        #y_q = y[mask]

        # Compute moving average for center line
        #y_mean = np.convolve(y_q, np.ones(window_size)/window_size, mode='valid')
        # Adjust x_q to match the length of y_mean (trim edges due to convolution)
        #trim = (window_size - 1) // 2
        #x_mean = x_q[trim:len(x_q)-trim]

        #if len(x_mean) <= 1:
        #    continue  # Skip if not enough points after trimming

        # Plot center line
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode='lines',
                name=f'{legend} Q{qmin:.2f}',
                line=dict(
                    color=colors[i],
                    width=LINE_WIDTH
                )
            )
        )

        # Plot scatter for points in this quantile range
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
                mode='markers',
                name=f'{legend} Q{qmin:.2f} Points',
                marker=dict(
                    color=colors[i],
                    size=5
                ),
                showlegend=False  # Hide scatter legend to avoid clutter
            )
        )

    # Update axes
    fig.update_xaxes(title_text=xlabel)
    if log_x:
        fig.update_xaxes(type='log')

    return fig

def pareto_points(steps, costs, scores, soft=0.0):
    pareto_steps = []
    pareto_costs = []
    pareto_scores = []
    max_score = max(scores)
    for i in range(len(steps)):
        #if scores[i] < 0.25*max_score:
        #    continue 

        better = [scores[j] >= scores[i] and 
            costs[j] < costs[i]*(1 - soft) and steps[j] < steps[i]*(1 - soft)
            for j in range(len(scores))]
        if not any(better):
            pareto_steps.append(steps[i])
            pareto_costs.append(costs[i])
            pareto_scores.append(scores[i])

    idxs = np.argsort(pareto_steps)
    pareto_steps = [pareto_steps[i] for i in idxs]
    pareto_costs = [pareto_costs[i] for i in idxs]
    pareto_scores = [pareto_scores[i] for i in idxs]
    return pareto_steps, pareto_costs, pareto_scores

def load_seed_data(filename):
    with open(filename, 'r') as f:
        experiments = json.load(f)

    all_uptime = []
    all_perf = []
    for trial in experiments:
        uptime = []
        perf = []
        for e in trial:
            u = e['uptime']
            if 'environment/perf' not in e:
                continue

            uptime.append(u)
            perf.append(e['environment/perf'])

        all_uptime.append(uptime)
        all_perf.append(perf)

    return all_uptime, all_perf

def load_hyper_data(filename):
    with open(filename, 'r') as f:
        experiments = json.load(f)

    all_hyper = []
    all_perf = []
    for trial in experiments:
        hyper = trial[-1]['learning_rate']
        perf = trial[-1]['environment/perf']
        all_hyper.append(hyper)
        all_perf.append(perf)

    all_hyper = np.array(all_hyper).reshape(3, -1)
    all_perf = np.array(all_perf).reshape(3, -1)
    return all_hyper, all_perf


def load_sweep_data(path):
    data = {}
    keys = None
    for fpath in glob.glob(path):
        with open(fpath, 'r') as f:
            exp = json.load(f)

        if not data:
            for kk in exp.keys():
                if kk == 'data':
                    for k, v in exp[kk][-1].items():
                        data[k] = []
                else:
                    data[kk] = []

        discard = False
        for kk in list(data.keys()):
            if kk not in exp and kk not in exp['data'][-1]:
                discard = True
                break

        if discard:
            continue

        for kk in list(data.keys()):
            if kk in exp:
                v = exp[kk]
                sweep_key = f'sweep/{kk}/distribution'
                if sweep_key in data and exp[sweep_key] == 'logit_normal':
                    v = 1 - v
                elif kk in ('train/vtrace_rho_clip', 'train/vtrace_c_clip'):
                    # Temporary hack for bad bounds
                    v = max(v, 0.1)

                data[kk].append(v)
            else:
                data[kk].append(exp['data'][-1][kk])

    return data

def cached_sweep_load(path):
    cache_file = os.path.join(path, 'cache.json')
    if not os.path.exists(cache_file):
        data = load_sweep_data(os.path.join(path, '*.json'))
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    with open(cache_file, 'r') as f:
        data = json.load(f)

    return data

    '''
    # Step 1: Check if cache exists; if not, create it using load_sweep_data
    if not os.path.exists(cache_file):
        experiments = load_sweep_data(os.path.join(path, '*.json'))
        # Create cache as list of [filename, data] pairs
        cache_data = [
            [os.path.basename(fpath), exp]
            for fpath, exp in zip(glob.glob(os.path.join(path, '*.json')), experiments)
            if not fpath.endswith('cache.json')  # Exclude cache file itself
        ]
        # Write cache
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    # Step 2: Load existing cache
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    # Convert cache to dict: {filename: data}
    cache_dict = {item[0]: item[1] for item in cache_data}

    # Get current files in directory (excluding cache.json)
    current_files = set(
        os.path.basename(f) for f in glob.glob(os.path.join(path, '*.json'))
        if not f.endswith('cache.json')
    )
    cached_files = set(cache_dict.keys())

    # Step 3: Check for new files not in cache
    new_files = current_files - cached_files
    if new_files:
        # Load only new files using a modified load_sweep_data
        new_experiments = []
        new_file_paths = [os.path.join(path, fname) for fname in new_files]
        for fpath in new_file_paths:
            with open(fpath, 'r') as f:
                exp = json.load(f)
            data = {}
            for kk, vv in exp.items():
                if kk == 'data':
                    for k, v in exp[kk][-1].items():
                        data[k] = v
                else:
                    data[kk] = vv
            new_experiments.append([os.path.basename(fpath), data])

        # Update cache with new experiments
        cache_data.extend(new_experiments)
        # Write updated cache
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    return [e[1] for e in cache_data]
    return cache_data

    # Rebuild cache_dict to include any new files
    cache_dict = {item[0]: item[1] for item in cache_data}

    # Return cached data as a dictionary
    return cache_dict
    '''


from statsmodels.nonparametric.smoothers_lowess import lowess
def compute_bin_stats(x, s, y, n_bins=20, overlap=0.5, score_threshold=0.95):
    """
    Bin data, select high-performing points, and compute weighted stats.
    x: steps (log-scaled compute/samples), s: scores, y: hyperparameter values
    """
    # Log-scale x to handle compute/samples range
    x_log = np.log10(x)
    x_min, x_max = x_log.min(), x_log.max()
    bin_width = (x_max - x_min) / (n_bins * (1 - overlap))
    bin_centers = np.linspace(x_min, x_max, n_bins)

    y_weighted = []
    y_lower = []
    y_upper = []

    for center in bin_centers:
        # Define bin boundaries with overlap
        bin_start = center - bin_width / 2
        bin_end = center + bin_width / 2
        mask = (x_log >= bin_start) & (x_log <= bin_end)

        # Select high-performing points (scores within 95% of max in bin)
        bin_s = s[mask]
        if len(bin_s) == 0:
            y_weighted.append(np.nan)
            y_lower.append(np.nan)
            y_upper.append(np.nan)
            continue
        s_max = bin_s.max()
        high_perf_mask = bin_s >= score_threshold * s_max

        # Compute weighted mean and quantiles for y
        bin_y = y[mask][high_perf_mask]
        bin_s = bin_s[high_perf_mask]
        if len(bin_y) == 0:
            y_weighted.append(np.nan)
            y_lower.append(np.nan)
            y_upper.append(np.nan)
            continue
        weights = bin_s / bin_s.sum()  # Normalize scores as weights
        y_mean = np.average(bin_y, weights=weights)
        y_quantiles = np.percentile(bin_y, [25, 75])  # IQR for stability range

        y_weighted.append(y_mean)
        y_lower.append(y_quantiles[0])
        y_upper.append(y_quantiles[1])

    return bin_centers, np.array(y_weighted), np.array(y_lower), np.array(y_upper)

def loess_fit(x, s, y, n_bins=20, frac=0.4):
    """
    Perform LOESS fit on binned data for smoothed curve and ribbons.
    """
    # Compute bin statistics
    bin_centers, y_weighted, y_lower, y_upper = compute_bin_stats(x, s, y, n_bins=n_bins)

    # Remove NaNs for LOESS
    valid_mask = ~np.isnan(y_weighted)
    bin_centers = bin_centers[valid_mask]
    y_weighted = y_weighted[valid_mask]
    y_lower = y_lower[valid_mask]
    y_upper = y_upper[valid_mask]

    # Apply LOESS to weighted mean, lower, and upper bounds
    smoothed_y = lowess(y_weighted, bin_centers, frac=frac, return_sorted=True)
    smoothed_lower = lowess(y_lower, bin_centers, frac=frac, return_sorted=True)
    smoothed_upper = lowess(y_upper, bin_centers, frac=frac, return_sorted=True)

    # Convert back to original x scale
    x_smooth = 10 ** smoothed_y[:, 0]  # Undo log-scale
    y_smooth = smoothed_y[:, 1]
    y_smooth_lower = smoothed_lower[:, 1]
    y_smooth_upper = smoothed_upper[:, 1]

    return x_smooth, y_smooth, y_smooth_lower, y_smooth_upper

#fig1 = figure(title='Hyperparameter Ablation', xlabel='Learning Rate', legend='Ablate', xaxis_type='log')
#all_hyper, all_perf = load_hyper_data('puffer_pong_learning_rate.npz')
#plot_group(fig1, all_hyper, all_perf, legend='Pong')
#all_hyper, all_perf = load_hyper_data('puffer_breakout_learning_rate.npz')
#plot_group(fig1, all_hyper, all_perf, legend='Breakout', i=1)

#fig2 = figure(title='Seed Sensitivity', xlabel='Uptime', legend='Ablate')
#all_uptime, all_perf = load_seed_data('puffer_pong_seeds.npz')
#plot_group(fig2, all_uptime, all_perf, legend='Pong')
#all_uptime, all_perf = load_seed_data('puffer_breakout_seeds.npz')
#plot_group(fig2, all_uptime, all_perf, legend='Breakout', i=1)
#all_uptime, all_perf = load_seed_data('puffer_connect4_seeds.npz')
#plot_group(fig2, all_uptime, all_perf, legend='Connect4', i=2)

env_names = ['breakout', 'pong']
EXPERIMENTS = {
    name: cached_sweep_load(f'experiments/logs/puffer_{name}')
    for name in env_names
}

# Initialize Dash app
app = Dash()
app.css.append_css({'external_stylesheets': 'dash.css'})
app.layout = html.Div([
    html.H1('The Puffer Frontier Project', style={'textAlign': 'center'}),
    html.Br(),
    html.Label([
        "Score Threshold %: ",
        dcc.Slider(
            id='pareto-slider',
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.95,
            marks={i: str(0.05*i) for i in range(0, 21)},
        )
    ]),
    dcc.Graph(id='pareto'),
    html.Br(),
    html.Label([
        "Score Threshold %: ",
        dcc.Slider(
            id='hyper-box-thresh',
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.95,
            marks={i: str(0.05*i) for i in range(0, 21)}
        )
    ]),
    html.Label([
        "Bins: ",
        dcc.Slider(
            id='hyper-box-buckets',
            min=1,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(0, 11)}
        )
    ]),
    html.Label([
        "X Axis: ",
        dcc.Dropdown(
            id="hyper-box-x",
            options=[{"label": key, "value": key} for key in ['cost', 'agent_steps']],
            value="agent_steps",
            style={"width": "50%"}
        )
    ]),
    dcc.Graph(id='hyper-box'),
    html.Br(),
    html.Label([
        "Score Threshold %: ",
        dcc.Slider(
            id='hyper-agg-slider',
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.95,
            marks={i: str(0.05*i) for i in range(0, 21)}
        )
    ]),
    html.Label([
        "Steps Interval: ",
        dcc.RangeSlider(
            id='hyper-agg-range',
            min=0.0,
            max=1.0,
            step=0.1,
            value=[0.0, 1.0]
        )
    ]),
    dcc.Graph(id='hyper-agg'),
    html.Br(),
    html.Label([
        "Environment: ",
        dcc.Dropdown(
            id="scatter-dropdown-env",
            options=[{"label": key, "value": key} for key in env_names],
            value="breakout",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "X: ",
        dcc.Dropdown(
            id="scatter-dropdown-x",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "Y: ",
        dcc.Dropdown(
            id="scatter-dropdown-y",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/score",
            style={"width": "50%"}
        )
    ]),
    dcc.Graph(id='scatter')
],
style={"width": 1280}
)

@app.callback(
    Output("pareto", "figure"),
    Input("pareto-slider", "value")
)
def update_pareto_plot(thresh):
    f = figure(title='Compute/Data Pareto Front', xlabel='Steps', ylabel='Cost', legend='Trial')

    for i, env in enumerate(EXPERIMENTS):
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

        # Header plot
        #if key == 'front':
        plot_quantiles(f, EXPERIMENTS[env], thresh, xlabel='Steps', legend=env, log_x=False, i=i)
        '''
        elif key == 'score':
            f = figure(title='Sweep', xlabel='Steps', ylabel='Scores', legend='Trial')
            scatter(f, steps, scores, costs, legend=env_name)
        elif key == 'cost':
            f = figure(title='Sweep', xlabel='Cost', ylabel='Scores', legend='Trial')
            scatter(f, costs, scores, steps, legend=env_name)
        '''

    return f

@app.callback(
    Output("scatter", "figure"),
    Input("scatter-dropdown-env", "value"),
    Input("scatter-dropdown-x", "value"),
    Input("scatter-dropdown-y", "value")
)
def update_scatter(env, xkey, ykey):
    steps = EXPERIMENTS[env]['agent_steps']
    costs = EXPERIMENTS[env]['cost']
    scores = EXPERIMENTS[env]['environment/score']

    # TODO: This is not applying frameskip 
    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS[env]:
        skip = EXPERIMENTS[env]['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    f = figure(title='Experiments', xlabel=xkey, ylabel=ykey, legend='Ablate')
    #f.update_yaxes(type='log')

    x = EXPERIMENTS[env][xkey]
    y = EXPERIMENTS[env][ykey]
    c = scores
    scatter(f, x, y, c, showlegend=False)

    '''
    # Filter by score
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > 0.95*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]

    f = figure(title='Hyper Stability', xlabel='Steps', ylabel='Hyper', legend='Ablate')
    f.update_yaxes(type='log')

    for j, hyper in enumerate(HYPERS):
        c = LINE_COLORS[j % len(LINE_COLORS)]
        s = [scores[i] for i in idxs]
        x = [steps[i] for i in idxs]
        y = [EXPERIMENTS[env][hyper][i] for i in idxs]

        x, y, s = pareto_points(x, y, s, 0.1)
        scatter(f, x, y, c, showlegend=False)

        x = np.array(x)
        s = np.array(s)
        y = np.array(y)
        x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
        s = np.ones_like(x_smooth)

        f.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=hyper,
                line=dict(
                    color=c,
                    width=LINE_WIDTH
                )
            )
         )
    '''

    return f


@app.callback(
    Output("hyper-stable", "figure"),
    Input("hyper-stable-slider", "value")
)
def update_hyper_stable(thresh):
    for i, env in enumerate(EXPERIMENTS):
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS[env]:
        skip = EXPERIMENTS[env]['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    # Filter by score
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > 0.95*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]

    f = figure(title='Hyper Stability', xlabel='Steps', ylabel='Hyper', legend='Ablate')
    f.update_yaxes(type='log')

    for j, hyper in enumerate(HYPERS):
        c = LINE_COLORS[j % len(LINE_COLORS)]
        s = [scores[i] for i in idxs]
        x = [steps[i] for i in idxs]
        y = [EXPERIMENTS[env][hyper][i] for i in idxs]

        x, y, s = pareto_points(x, y, s, 0.1)
        scatter(f, x, y, c, showlegend=False)

        x = np.array(x)
        s = np.array(s)
        y = np.array(y)
        x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
        s = np.ones_like(x_smooth)

        f.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=hyper,
                line=dict(
                    color=c,
                    width=LINE_WIDTH
                )
            )
         )

    return f

@app.callback(
    Output("hyper-box", "figure"),
    Input("hyper-box-thresh", "value"),
    Input("hyper-box-buckets", "value"),
    Input("hyper-box-x", "value")
)
def update_hyper_box(thresh, buckets, x):
    # Initialize data storage
    env_data = {}

    # Process each environment
    for env in EXPERIMENTS:
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

        # Adjust steps if frameskip exists
        if 'env/frameskip' in EXPERIMENTS[env]:
            skip = EXPERIMENTS[env]['env/frameskip']
            steps = [n*m for n, m in zip(steps, skip)]

        # Filter by score threshold
        max_score = max(scores)
        idxs = [i for i, s in enumerate(scores) if s > thresh*max_score]

        # Select x-axis data based on input
        x_data = costs if x == 'cost' else steps
        filtered_x = [x_data[i] for i in idxs]

        # Get all hyperparameters

        # Store filtered data for this environment
        hyper_data = {}
        env_data[env] = {'x': filtered_x, 'hypers': hyper_data}
        for h in HYPERS:
            hyper_data[h] = [EXPERIMENTS[env][h][i] for i in idxs]

    # Create buckets
    all_x = [x for env in env_data for x in env_data[env]['x']]
    x_min, x_max = min(all_x), max(all_x)
    bucket_edges = np.linspace(x_min, x_max, buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2

    # Initialize heatmap data
    heatmap_data = np.zeros((len(HYPERS), buckets))

    # Compute means for each bucket and hyperparameter
    for i, hyper in enumerate(HYPERS):
        for j in range(buckets):
            bucket_means = []
            for env in env_data:
                if hyper not in env_data[env]['hypers']:
                    continue

                x_vals = np.array(env_data[env]['x'])
                hyper_vals = np.array(env_data[env]['hypers'][hyper])
                # Find indices in current bucket
                idxs = (x_vals >= bucket_edges[j]) & (x_vals < bucket_edges[j+1])
                if np.any(idxs):
                    bucket_means.append(np.mean(hyper_vals[idxs]))

            # Average across environments
            heatmap_data[i, j] = np.mean(bucket_means) if bucket_means else np.nan

    heatmap_data = np.log(heatmap_data)

    # Create heatmap
    f = figure(title="Hyperparameter Drift",
        xlabel=x.capitalize(),
        ylabel="Hyperparameters"
    )

    f.add_trace(
        go.Heatmap(
            x=bucket_centers,
            y=HYPERS,
            z=heatmap_data,
            colorscale='Viridis',
            showscale=True,
            zmin=np.nanmin(heatmap_data),
            zmax=np.nanmax(heatmap_data),
            colorbar=dict(title="Value")
        )
    )

    return f

@app.callback(
    Output("hyper", "figure"),
    Input("hyper-dropdown", "value")
)
def update_hyper_plot(hyper):
    for i, env in enumerate(EXPERIMENTS):
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS[env]:
        skip = EXPERIMENTS[env]['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    # Filter by score
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > 0.95*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]

    f = figure(title=hyper, xlabel='Steps', ylabel='Hyper', legend='Ablate')
    y = [EXPERIMENTS[env][hyper][i] for i in idxs]
    s = [scores[i] for i in idxs]
    x = [steps[i] for i in idxs]

    x, y, s = pareto_points(x, y, s, 0.1)
    scatter(f, x, y, s, legend=env_name)

    x = np.array(x)
    s = np.array(s)
    y = np.array(y)
    x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
    s = np.ones_like(x_smooth)

    plot_lines(f, [x_smooth], [y_smooth])
    return f

from plotly import graph_objects as go
from dash import Output, Input

@app.callback(
    Output("hyper-agg", "figure"),
    Input("hyper-agg-slider", "value"),
    Input("hyper-agg-range", "value")
)
def update_hyper_agg_plot(thresh, step_range):
    # Initialize figure
    f = go.Figure()
    f.update_layout(
        title=dict(text='Hyperparameter Stable Range', font=TITLE_FONT),
        xaxis=dict(title=dict(text='Value', font=AXIS_FONT), tickfont=TICK_FONT),
        yaxis=dict(title=dict(text='Hyper', font=AXIS_FONT), tickfont=TICK_FONT),
        showlegend=True,
        legend=dict(font=LEGEND_FONT),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        width=1280,
        height=720,
        autosize=False,
        xaxis_type='log',
        barmode='overlay',  # Overlay bars instead of stacking
    )
    f.update_xaxes(showgrid=False)
    f.update_yaxes(showgrid=False)

    for i, env in enumerate(EXPERIMENTS):
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

        # Adjust steps
        if 'env/frameskip' in EXPERIMENTS[env]:
            skip = EXPERIMENTS[env]['env/frameskip']
            steps = [n*m for n, m in zip(steps, skip)]

        max_score = max(scores)
        max_steps = max(steps)
        n = len(scores)
        idxs = [i for i in range(n) if scores[i] > thresh*max_score and
            step_range[0]<steps[i]/max_steps<step_range[1]]

        if len(idxs) < 2:
            continue
        
        for k, hyper in enumerate(HYPERS):
            y = [EXPERIMENTS[env][hyper][i] for i in idxs]
            ymin = min(y)
            ymax = max(y)
            f.add_trace(
                go.Bar(
                    x=[ymax - ymin],
                    y=[hyper],  # Hyperparameter as x-axis
                    base=ymin,
                    showlegend=False,
                    marker_color='blue',
                    opacity=0.5,
                    width=1.0,
                    orientation='h'
                )
            )

    return f

# Set layout with static graph
#app.layout = layout
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
