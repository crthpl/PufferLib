from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import glob
import os

FONT_FAMILY = 'Arial'
FONT_SIZE_TITLE = 28
FONT_SIZE_AXIS = 22
FONT_SIZE_TICK = 20
FONT_SIZE_TICK_3D = 14
FONT_SIZE_LEGEND = 18
FONT_COLOR = '#f1f1f1'
PLOT_BG_COLOR = '#061a1a'
PAPER_BG_COLOR = '#061a1a'
LINE_WIDTH = 4
LINE_COLORS = ["#0000b3", "#0010d9", "#0020ff", "#0040ff", "#0060ff", "#0080ff", "#009fff", "#00bfff", "#00ffff"][::-1]
roygbiv = np.random.permutation(['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'])
#roygbiv = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
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
GRID_COLOR = '#00f1f1'
TICK_FONT_3D = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TICK_3D,
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
    'env/num_envs',
]
ALL_KEYS = [
    'agent_steps',
    'cost',
    'environment/score',
    'environment/perf'
] + HYPERS

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
                    v = max(v, 0.1)

                data[kk].append(v)
            else:
                data[kk].append(exp['data'][-1][kk])

    return data

def cached_sweep_load(path, env_name):
    cache_file = os.path.join(path, 'cache.json')
    if not os.path.exists(cache_file):
        data = load_sweep_data(os.path.join(path, '*.json'))
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    with open(cache_file, 'r') as f:
        data = json.load(f)

    steps = data['agent_steps']
    costs = data['cost']
    scores = data['environment/score']

    idxs = pareto_idx(steps, costs, scores)
    
    # Create a DataFrame for this environment
    df_data = {}
    for k in data:
        df_data[k] = [data[k][i] for i in idxs]
    
    # Apply performance cap
    df_data['environment/perf'] = [min(e, 1.0) for e in df_data['environment/perf']]
    
    # Adjust steps by frameskip if present
    if 'env/frameskip' in df_data:
        skip = df_data['env/frameskip']
        df_data['agent_steps'] = [n*m for n, m in zip(df_data['agent_steps'], skip)]
    
    # Add environment name
    df_data['env_name'] = [env_name] * len(idxs)
    
    return pd.DataFrame(df_data)

env_names = ['tripletriad', 'grid', 'moba', 'tower_climb', 'tetris', 'breakout', 'pong', 'g2048', 'snake', 'pacman']
#env_names = ['grid', 'breakout', 'g2048']
#env_names = ['grid']

# Create a list of DataFrames for each environment
dfs = [cached_sweep_load(f'experiments/logs/puffer_{name}', name) for name in env_names]

# Concatenate all DataFrames into a single DataFrame
EXPERIMENTS = pd.concat(dfs, ignore_index=True)
EXPERIMENTS.set_index('env_name', inplace=True)

app = Dash()
app.css.append_css({'external_stylesheets': 'dash.css'})
app.layout = html.Div([
    html.H1('Puffer Constellation', style={'textAlign': 'center'}),
    html.Br(),

    html.Label([
        "X: ",
        dcc.Dropdown(
            id="optimal-dropdown-x",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="cost",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "Y: ",
        dcc.Dropdown(
            id="optimal-dropdown-y",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "Z: ",
        dcc.Dropdown(
            id="optimal-dropdown-z",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/perf",
            style={"width": "50%"}
        )
    ]),
    dcc.Graph(id='optimal'),
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
            value="cost",
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
    html.Label([
        "Z: ",
        dcc.Dropdown(
            id="scatter-dropdown-z",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        )
    ]),
    dcc.Graph(id='scatter'),
    html.Br(),

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
        "Threshold: ",
        dcc.Slider(
            id='pca-slider',
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.5,
            marks={i: str(0.05*i) for i in range(0, 21)}
        )
    ]),
    dcc.Graph(id='pca'),

],
style={"width": 1280}
)

@app.callback(
    Output("optimal", "figure"),
    Input("optimal-dropdown-x", "value"),
    Input("optimal-dropdown-y", "value"),
    Input("optimal-dropdown-z", "value")
)
def update_optimal_plot(xkey, ykey, zkey):
    all_x = []
    all_y = []
    all_z = []
    all_env = []
    for env in env_names:
        env_data = EXPERIMENTS.loc[env]
        all_x.append(env_data[xkey].copy())
        all_y.append(env_data[ykey].copy())
        all_z.append(env_data[zkey].copy())
        all_env += [env] * len(env_data[xkey])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    f = px.scatter_3d(x=all_x, y=all_y, z=all_z, color=all_env, log_x=True, log_y=True, log_z=False, color_discrete_sequence=roygbiv)
    layout_dict = {
        'title': dict(text='Pareto', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'scene': dict(
            xaxis=dict(
                title=dict(text=xkey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='log',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text=ykey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='log',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            zaxis=dict(
                title=dict(text=zkey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='linear',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            bgcolor=PLOT_BG_COLOR,
        )
    }
    f.update_layout(**layout_dict)
    return f


@app.callback(
    Output("scatter", "figure"),
    Input("scatter-dropdown-env", "value"),
    Input("scatter-dropdown-x", "value"),
    Input("scatter-dropdown-y", "value"),
    Input("scatter-dropdown-z", "value")
)
def update_scatter(env, xkey, ykey, zkey):
    env_data = EXPERIMENTS.loc[env]
    x = env_data[xkey]
    y = env_data[ykey]
    z = env_data[zkey]
    mmin = min(z)
    mmax = max(z)
    thresh = np.linspace(mmin, mmax, 8)
    all_fx = []
    all_fy = []
    bin_label = []
    for j in range(7):
        idxs = [i for i, e in enumerate(z) if thresh[j] < e < thresh[j+1]]
        if len(idxs) <= 2:
            continue
        fx = [x[i] for i in idxs]
        fy = [y[i] for i in idxs]
        all_fx += fx
        all_fy += fy
        bin_label += [str(thresh[j])] * len(fx)
    f = px.scatter(x=all_fx, y=all_fy, color=bin_label, color_discrete_sequence=roygbiv)
    f.update_traces(marker_size=10)
    layout_dict = {
        'title': dict(text='Experiments', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text=xkey, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        ),
        'yaxis': dict(
            title=dict(text=ykey, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        )
    }
    f.update_layout(**layout_dict)
    return f

@app.callback(
    Output("hyper-box", "figure"),
    Input("hyper-box-x", "value")
)
def update_hyper_box(x):
    buckets = 4
    env_data = {}
    for env in env_names:
        data = EXPERIMENTS.loc[env]
        steps = data['agent_steps']
        costs = data['cost']
        scores = data['environment/score']
        x_data = costs if x == 'cost' else steps
        hyper_data = {}
        env_data[env] = {'x': x_data, 'hypers': hyper_data}
        for h in HYPERS:
            hyper_data[h] = data[h]
    all_x = [x for env in env_data for x in env_data[env]['x']]
    x_min, x_max = min(all_x), max(all_x)
    bucket_edges = np.linspace(x_min, x_max, buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    heatmap_data = np.zeros((len(HYPERS), buckets))
    for i, hyper in enumerate(HYPERS):
        for j in range(buckets):
            bucket_means = []
            for env in env_data:
                if hyper not in env_data[env]['hypers']:
                    continue
                x_vals = np.array(env_data[env]['x'])
                hyper_vals = np.array(env_data[env]['hypers'][hyper])
                idxs = (x_vals >= bucket_edges[j]) & (x_vals < bucket_edges[j+1])
                if np.any(idxs):
                    bucket_means.append(np.mean(hyper_vals[idxs]))
            heatmap_data[i, j] = np.mean(bucket_means) if bucket_means else np.nan
    heatmap_data = np.log(heatmap_data)
    heatmap_data -= heatmap_data[:, 0, None] # Normalize
    f = px.imshow(heatmap_data, x=bucket_centers, y=HYPERS, color_continuous_scale='Viridis', zmin=np.nanmin(heatmap_data), zmax=np.nanmax(heatmap_data), labels=dict(color="Value"))
    layout_dict = {
        'title': dict(text="Hyperparameter Drift", font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text=x.capitalize(), font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        ),
        'yaxis': dict(
            title=dict(text="Hyperparameters", font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        )
    }
    f.update_layout(**layout_dict)
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

    for i, env in enumerate(env_names):
        env_data = EXPERIMENTS.loc[env]
        steps = env_data['agent_steps']
        costs = env_data['cost']
        scores = env_data['environment/score']

        max_score = max(scores)
        max_steps = max(steps)
        n = len(scores)
        idxs = [i for i in range(n) if scores[i] > thresh*max_score and
            step_range[0]<steps[i]/max_steps<step_range[1]]

        if len(idxs) < 2:
            continue
        
        for k, hyper in enumerate(HYPERS):
            y = [env_data[hyper][i] for i in idxs]

            ymin = min(y)
            ymax = max(y)
            f.add_trace(
                go.Bar(
                    x=[ymax - ymin],
                    y=[hyper],  # Hyperparameter as x-axis
                    base=ymin,
                    showlegend=False,
                    marker_color='#00f1f1',
                    opacity=0.25,
                    width=1.0,
                    orientation='h'
                )
            )

    return f

@app.callback(
    Output("pca", "figure"),
    Input("pca-slider", "value"),
)
def update_pca_plot(thresh):
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

    filtered = {env: [] for env in env_names}
    for env in env_names:
        env_data = EXPERIMENTS.loc[env]
        perf = env_data['environment/perf']
        idxs = [i for i in range(len(perf)) if perf[i] > thresh]
        for hyper in HYPERS:
            filt = np.array([env_data[hyper][i] for i in idxs])
            mmin = np.array(env_data[f'sweep/{hyper}/min'])
            mmin = [mmin[i] for i in idxs]
            mmax = env_data[f'sweep/{hyper}/max']
            mmax = np.array([mmax[i] for i in idxs])
            distribution = env_data[f'sweep/{hyper}/distribution'][0]
            #if 'uniform' in distribution:
            #    #filt = (filt - mmin) / (mmax - mmin)
            #    pass
            if 'log' in distribution or 'pow2' in distribution:
                filt = np.log(filt)
                #filt = (np.log(filt) - np.log(mmin)) / (np.log(mmax) - np.log(mmin))

            filtered[env].append(filt)

        filtered[env] = np.array(filtered[env]).T

    training = np.concatenate(list(filtered.values()), axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(training)

    all_x = []
    all_y = []
    all_z = []
    for i, env in enumerate(filtered):
        if filtered[env].shape[0] == 0:
            continue

        reduced = pca.transform(filtered[env])
        x, y = reduced[:, 0], reduced[:, 1]
        all_x.append(x)
        all_y.append(y)
        all_z.append([env]*len(x))

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    f = px.scatter(x=all_x, y=all_y, color=all_z, color_discrete_sequence=roygbiv)
 
    f.update_traces(marker_size=10)
    layout_dict = {
        'title': dict(text='Experiments', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text='principal component 1', font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        ),
        'yaxis': dict(
            title=dict(text='principal component 2', font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        )
    }
    f.update_layout(**layout_dict)
    return f


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
