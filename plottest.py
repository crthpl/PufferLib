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

def rgba(hex, alpha):
    return f"rgba({int(hex[1:3], 16)}, {int(hex[3:5], 16)}, {int(hex[5:7], 16)}, {alpha})"

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

def plot_lines(fig, xx, yy):
    for i, (x, y) in enumerate(zip(xx, yy)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'Trial {i+1}',
                line=dict(
                    color=LINE_COLORS[i % len(LINE_COLORS)],
                    width=LINE_WIDTH
                )
            )
         )

def scatter(fig, x, y, c, legend='Trial', log_x=False, i=0):
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

def plot_quantiles(fig, x, y, s, xlabel='Performance', legend='Trial', log_x=False, i=0):
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    # Sort data by x for smooth plotting
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    s = s[sort_idx]

    # Define quantile thresholds (in descending order for proper range filtering)
    quantile_thresholds = [0.95, 0.9, 0.75, 0.5, 0.25, 0.0]
    quantile_thresholds = [e*s.max() for e in quantile_thresholds]
    colors = ['blue', 'green', 'orange', 'red']

    # Plot center lines and scatter for each quantile range
    for j, (q, color) in enumerate(zip(quantile_thresholds, colors)):
        # Define the range for this quantile
        if j == 0:
            # Highest quantile: s >= q
            mask = s >= q
        else:
            # Other quantiles: q <= s < previous_q
            prev_q = quantile_thresholds[j - 1]
            mask = (s >= q) & (s < prev_q)

        if np.sum(mask) <= 5:
            continue  # Skip

        fx = x[mask]
        fy = y[mask]
        fs = np.ones_like(fx) # More robust to bin scores into quantiles

        #px, py, ps = pareto_points(fx, fy, fs)
        fx, fy, fs = pareto_points(fx, fy, fs, 0.1)

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
                name=f'{legend} Q{q:.2f}',
                line=dict(
                    color=color,
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
                name=f'{legend} Q{q:.2f} Points',
                marker=dict(
                    color=color,
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
                data[kk].append(exp[kk])
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

env_name = 'breakout'
EXPERIMENTS = cached_sweep_load(f'experiments/logs/puffer_{env_name}')

# Initialize Dash app
app = Dash()
app.layout = html.Div([
    html.H1('The Puffer Frontier Project', style={'textAlign': 'center'}),
    #dcc.Graph(figure=step_cost),
    #dcc.Graph(figure=step_score),
    #dcc.Graph(figure=cost_score),
    html.Br(),
    dcc.Dropdown(
        id="pareto-dropdown",
        options=[
            {"label": 'front', "value": 'front'},
            {"label": 'cost', "value": 'cost'},
            {"label": 'score', "value": 'score'},
        ],
        value="front",
        style={"width": "50%"}
    ),
    dcc.Graph(id='pareto'),
    dcc.Slider(
        id='hyper-agg-slider',
        min=0.0,
        max=1.0,
        step=0.05,
        value=0.95,
        marks={i: str(0.05*i) for i in range(0, 21)}
    ),
    dcc.Graph(id='hyper-agg'),
    dcc.Dropdown(
        id="hyper-dropdown",
        options=[{"label": key, "value": key} for key in HYPERS],
        value="train/learning_rate",
        style={"width": "50%"}
    ),
    dcc.Graph(id='hyper')
])

@app.callback(
    Output("pareto", "figure"),
    Input("pareto-dropdown", "value")
)
def update_pareto_plot(key):
    steps = EXPERIMENTS['agent_steps']
    costs = EXPERIMENTS['cost']
    scores = EXPERIMENTS['environment/score']

    # Filter outliers
    '''
    idxs = [i for i, s in enumerate(steps) if s < 1e6]
    experiments = [experiments[i] for i in idxs]
    steps = [steps[i] for i in idxs]
    costs = [costs[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    '''

    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS:
        skip = EXPERIMENTS['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    # Filter by score
    '''
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > 0.95*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]
    '''

    #filtered_steps, filtered_costs, filtered_scores = pareto_points(steps, costs, scores)

    # Header plot
    if key == 'front':
        f = figure(title='Sweep', xlabel='Steps', ylabel='Cost', legend='Trial')
        plot_quantiles(f, steps, costs, scores,
            xlabel='Steps', legend='Trial', log_x=False, i=0)
        #scatter(f, filtered_steps, filtered_costs, filtered_scores, legend=env_name)
    elif key == 'score':
        f = figure(title='Sweep', xlabel='Steps', ylabel='Scores', legend='Trial')
        scatter(f, steps, scores, costs, legend=env_name)
    elif key == 'cost':
        f = figure(title='Sweep', xlabel='Cost', ylabel='Scores', legend='Trial')
        scatter(f, costs, scores, steps, legend=env_name)

    return f

    figs = []

    f = figure(title=hyper, xlabel='Steps', ylabel='Hyper', legend='Ablate')
    #idxs = [i for i, e in enumerate(experiments) if hyper in e]
    y = [EXPERIMENTS[i][hyper] for i in idxs]
    s = [scores[i] for i in idxs]
    #ss = [np.log(steps[i]) for i in idxs]
    x = [steps[i] for i in idxs]
    #c = [costs[i] for i in idxs]

    scatter(f, x, y, s, legend=env_name)

    x = np.array(x)
    s = np.array(s)
    y = np.array(y)
    x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
    s = np.ones_like(x_smooth)
    scatter(f, x_smooth, y_smooth, 'red', legend=env_name)

    #scatter(f, v, s, ss, legend=env_name)
    #scatter(f, x, y, s, legend=env_name)
    #figs.append(f)

    #pareto_steps, pareto_costs, pareto_scores = pareto_points(steps, costs, scores)
    #plot_lines(fig3, [pareto_steps], [pareto_costs])
    return f


    df = data_options[selected_dataset]
    fig = px.scatter(df, x="x", y="y", title=f"Scatter Plot for {selected_dataset}")
    return fig


@app.callback(
    Output("hyper", "figure"),
    Input("hyper-dropdown", "value")
)
def update_hyper_plot(hyper):
    steps = EXPERIMENTS['agent_steps']
    costs = EXPERIMENTS['cost']
    scores = EXPERIMENTS['environment/score']

    # Filter outliers
    '''
    idxs = [i for i, s in enumerate(steps) if s < 1e6]
    experiments = [experiments[i] for i in idxs]
    steps = [steps[i] for i in idxs]
    costs = [costs[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    '''

    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS:
        skip = EXPERIMENTS['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    # Filter by score
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > 0.95*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]

    # Header plot
    '''
    step_cost = figure(title='Sweep', xlabel='Steps', ylabel='Cost', legend='Trial')
    scatter(step_cost, filtered_steps, filtered_costs, filtered_scores, legend=env_name)

    step_score = figure(title='Sweep', xlabel='Steps', ylabel='Scores', legend='Trial')
    scatter(step_score, steps, scores, costs, legend=env_name)

    cost_score = figure(title='Sweep', xlabel='Cost', ylabel='Scores', legend='Trial')
    scatter(cost_score, costs, scores, steps, legend=env_name)
    '''

    figs = []

    f = figure(title=hyper, xlabel='Steps', ylabel='Hyper', legend='Ablate')
    #idxs = [i for i, e in enumerate(experiments) if hyper in e]
    y = [EXPERIMENTS[hyper][i] for i in idxs]
    s = [scores[i] for i in idxs]
    #ss = [np.log(steps[i]) for i in idxs]
    x = [steps[i] for i in idxs]
    #c = [costs[i] for i in idxs]

    scatter(f, x, y, s, legend=env_name)

    x = np.array(x)
    s = np.array(s)
    y = np.array(y)
    x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
    s = np.ones_like(x_smooth)

    plot_lines(f, [x_smooth], [y_smooth])
    #scatter(f, x_smooth, y_smooth, 'red', legend=env_name)

    #scatter(f, v, s, ss, legend=env_name)
    #scatter(f, x, y, s, legend=env_name)
    #figs.append(f)

    #pareto_steps, pareto_costs, pareto_scores = pareto_points(steps, costs, scores)
    #plot_lines(fig3, [pareto_steps], [pareto_costs])
    return f


    df = data_options[selected_dataset]
    fig = px.scatter(df, x="x", y="y", title=f"Scatter Plot for {selected_dataset}")
    return fig

@app.callback(
    Output("hyper-agg", "figure"),
    Input("hyper-agg-slider", "value")
)
def update_hyper_agg_plot(thresh):
    steps = EXPERIMENTS['agent_steps']
    costs = EXPERIMENTS['cost']
    scores = EXPERIMENTS['environment/score']

    # Filter outliers
    '''
    idxs = [i for i, s in enumerate(steps) if s < 1e6]
    experiments = [experiments[i] for i in idxs]
    steps = [steps[i] for i in idxs]
    costs = [costs[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    '''

    # Adjust steps
    if 'env/frameskip' in EXPERIMENTS:
        skip = EXPERIMENTS['env/frameskip']
        steps = [n*m for n, m in zip(steps, skip)]

    # Filter by score
    max_score = max(scores)
    idxs = [i for i, s in enumerate(scores) if s > thresh*max_score]
    filtered_steps = [steps[i] for i in idxs]
    filtered_costs = [costs[i] for i in idxs]
    filtered_scores = [scores[i] for i in idxs]

    # Header plot
    '''
    step_cost = figure(title='Sweep', xlabel='Steps', ylabel='Cost', legend='Trial')
    scatter(step_cost, filtered_steps, filtered_costs, filtered_scores, legend=env_name)

    step_score = figure(title='Sweep', xlabel='Steps', ylabel='Scores', legend='Trial')
    scatter(step_score, steps, scores, costs, legend=env_name)

    cost_score = figure(title='Sweep', xlabel='Cost', ylabel='Scores', legend='Trial')
    scatter(cost_score, costs, scores, steps, legend=env_name)
    '''

    figs = []

    f = figure(title='bar', xlabel='Steps', ylabel='Hyper', legend='Ablate')
    f.update_yaxes(type='log')
    for hyper in HYPERS:
        #idxs = [i for i, e in enumerate(experiments) if hyper in e]
        y = [EXPERIMENTS[hyper][i] for i in idxs]
        s = [scores[i] for i in idxs]
        #ss = [np.log(steps[i]) for i in idxs]
        x = [steps[i] for i in idxs]
        #c = [costs[i] for i in idxs]


        ymin = min(y)
        ymax = max(y)
        f.add_trace(
            go.Bar(
                x=[0],
                y=[ymax - ymin],
                base=ymin,
                name=hyper
            )
        )

    #scatter(f, x, y, s, legend=env_name)

    x = np.array(x)
    s = np.array(s)
    y = np.array(y)
    x_smooth, y_smooth, y_lower, y_upper = loess_fit(x, s, y, n_bins=20, frac=0.4)
    s = np.ones_like(x_smooth)

    #plot_lines(f, [x_smooth], [y_smooth])
    #scatter(f, x_smooth, y_smooth, 'red', legend=env_name)

    #scatter(f, v, s, ss, legend=env_name)
    #scatter(f, x, y, s, legend=env_name)
    #figs.append(f)

    #pareto_steps, pareto_costs, pareto_scores = pareto_points(steps, costs, scores)
    #plot_lines(fig3, [pareto_steps], [pareto_costs])
    return f


    df = data_options[selected_dataset]
    fig = px.scatter(df, x="x", y="y", title=f"Scatter Plot for {selected_dataset}")
    return fig



# Set layout with static graph
#app.layout = layout
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
