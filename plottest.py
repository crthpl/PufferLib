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

def scatter(fig, x, y, c, legend='Trial', log_x=False, i=0, showlegend=True):
    mmin = min(c)
    mmax = max(c)

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

def plot_quantiles(fig, experiments, thresh, xlabel='Performance', legend='Trial', log_x=False, i=0):
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
        s = filtered['environment/score']

        s_one = np.ones_like(x)
        fx, fy, fs = pareto_points(x, y, s_one, 0.00)

        # Plot center line
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
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
                x=x,
                y=y,
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
    f = figure(title='Compute/Data Pareto Front',
        xlabel='Steps', ylabel='Cost', legend='Trial')

    for i, env in enumerate(EXPERIMENTS):
        steps = EXPERIMENTS[env]['agent_steps']
        costs = EXPERIMENTS[env]['cost']
        scores = EXPERIMENTS[env]['environment/score']

        # Header plot
        plot_quantiles(f, EXPERIMENTS[env], thresh,
            xlabel='Steps', legend=env, log_x=False, i=i)

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

    x = EXPERIMENTS[env][xkey]
    y = EXPERIMENTS[env][ykey]
    c = scores
    scatter(f, x, y, c, showlegend=False)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
