import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> tuple[dict, dict]:
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads: dict, tails: dict, house_idx: dict, m: int) -> dict:
    """
    Find the swapped time series pair

    Parameters
    ---------
    heads: heads of time series
    tails: tails of time series
    house_idx: indices of houses
    m: subsequence length

    Returns
    --------
    min_score: time series pair with minimum swap-score
    """

    eps = 0.001

    # результат
    min_score = {
        'score': np.inf,
        'i': None,
        'j': None,
        'mp_j': None,   # mp для пары (H_i, T_j) с минимальным score
    }

    # маленький хелпер: привести всё к 1D numpy-вектору
    def to_1d(ts):
        # pandas DataFrame/Series
        if hasattr(ts, "to_numpy"):
            return ts.to_numpy().squeeze()
        # на всякий случай
        return np.asarray(ts).squeeze()

    # 1. считаем "нормальные" профили Head_i vs Tail_i (знаменатель)
    denom = {}  # i -> (min_dist, mp_i)
    for i in house_idx:
        head_i = to_1d(heads[f'H_{i}'])
        tail_i = to_1d(tails[f'T_{i}'])

        mp_i = compute_mp(ts1=head_i, ts2=tail_i, m=m)
        denom_i = np.min(mp_i['mp'])
        denom[i] = (denom_i, mp_i)

    # 2. перебираем все пары (i, j), j != i
    for i in house_idx:
        denom_i, _ = denom[i]

        for j in house_idx:
            if i == j:
                continue

            head_i = to_1d(heads[f'H_{i}'])
            tail_j = to_1d(tails[f'T_{j}'])

            mp_j = compute_mp(ts1=head_i, ts2=tail_j, m=m)
            num_ij = np.min(mp_j['mp'])

            swap_score_ij = num_ij / (denom_i + eps)

            if swap_score_ij < min_score['score']:
                min_score['score'] = swap_score_ij
                min_score['i'] = i
                min_score['j'] = j
                min_score['mp_j'] = mp_j
    
    return min_score


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")
