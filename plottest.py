from dash import Dash, html, dcc
import plotly.graph_objects as go
import numpy as np
import json

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

    colors = []
    for e in c:
        if e > 0.95*mmax:
            v = 1.0
        elif e > 0.9*mmax:
            v = 0.6
        elif e > 0.5*mmax:
            v = 0.3
        else:
            v = 0

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

def pareto_points(steps, costs, scores):
    pareto_steps = []
    pareto_costs = []
    pareto_scores = []
    max_score = max(scores)
    for i in range(len(steps)):
        if scores[i] < 0.25*max_score:
            continue 

        higher_score = [s for s in scores if s > scores[i]]
        lower_steps = [s for s in steps if s < scores[i]]
        lower_cost = [c for c in costs if c < costs[i]]
        better = [scores[j] > scores[i] and 
            costs[j] < costs[i] and steps[j] < steps[i]
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
    import glob
    costs = []
    steps = []
    scores = []
    for fpath in glob.glob(path):
        with open(fpath, 'r') as f:
            exp = json.load(f)

        cost = exp['cost']
        #step = exp['total_timesteps']
        step = exp['data'][-1]['agent_steps']
        score = exp['data'][-1]['environment/score']
        costs.append(cost)
        steps.append(step)
        scores.append(score)

    return steps, costs, scores


def layout():
    fig1 = figure(title='Hyperparameter Ablation', xlabel='Learning Rate', legend='Ablate', xaxis_type='log')
    #all_hyper, all_perf = load_hyper_data('puffer_pong_learning_rate.npz')
    #plot_group(fig1, all_hyper, all_perf, legend='Pong')
    #all_hyper, all_perf = load_hyper_data('puffer_breakout_learning_rate.npz')
    #plot_group(fig1, all_hyper, all_perf, legend='Breakout', i=1)

    fig2 = figure(title='Seed Sensitivity', xlabel='Uptime', legend='Ablate')
    #all_uptime, all_perf = load_seed_data('puffer_pong_seeds.npz')
    #plot_group(fig2, all_uptime, all_perf, legend='Pong')
    #all_uptime, all_perf = load_seed_data('puffer_breakout_seeds.npz')
    #plot_group(fig2, all_uptime, all_perf, legend='Breakout', i=1)
    #all_uptime, all_perf = load_seed_data('puffer_connect4_seeds.npz')
    #plot_group(fig2, all_uptime, all_perf, legend='Connect4', i=2)

    #fig3 = figure(title='Sweep', xlabel='Steps', ylabel='Cost', legend='Trial')
    fig3 = figure(title='Sweep', xlabel='Steps', ylabel='Score', legend='Trial')
    steps, costs, scores = load_sweep_data('experiments/logs/puffer_pong/*.json')
    #pareto_steps, pareto_costs, pareto_scores = pareto_points(steps, costs, scores)
    #plot_lines(fig3, [pareto_steps], [pareto_costs])
    #scatter(fig3, steps, costs, scores, legend='Pong')
    scatter(fig3, steps, scores, costs, legend='Pong')
    layout = html.Div([
        html.H1('The Puffer Frontier Project', style={'textAlign': 'center'}),
        dcc.Graph(figure=fig1),
        html.Br(),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])
    return layout



# Initialize Dash app
app = Dash()

# Set layout with static graph
app.layout = layout
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
