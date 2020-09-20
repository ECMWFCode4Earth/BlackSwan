# -------------------------------------------------------------------------
# IMPORTS

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.cm as cm
warnings.filterwarnings("ignore")
import math

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure
from math import radians
from pytz import timezone


# -------------------------------------------------------------------------
# FUNCTIONS

def initialize():
    with open('model_set.json') as f:
        model_set = json.load(f)
    return model_set['DAGMM']['train_end']

def read_tsdt():
    ROOT_PATH = "data/June30_mars_4min/"
    file = 'ts_bytes_database=marsod_240s.txt'
    raw_ts = pd.read_csv(ROOT_PATH + file, header=None).values.reshape(-1, )
    # raw_ts = raw_ts[:] / (np.mean(raw_ts) * 10)
    time_strings = pd.read_csv(ROOT_PATH + "dt.txt", header = None).values[:,0]
    raw_dt = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in time_strings]
    return raw_ts, raw_dt

def read_preds():
    PREDS_PATH = "predictions/DAGMM/sc2.txt"
    try:
        preds = pd.read_csv(PREDS_PATH, header=None).values.reshape(-1, )
        preds = preds[:] / (np.mean(preds) * 10)
    except:
        preds = []
    return preds

def update_data():
    global raw_ts
    global raw_dt
    global init_idx
    global cur_idx
    
    preds = read_preds()
    while(cur_idx < len(preds)):
        cur_ts = raw_ts[cur_idx + init_idx]
        cur_dt = raw_dt[cur_idx + init_idx]
        cur_sc = preds[cur_idx] * 1e12
        cur_idx += 1

        new_data = dict(dt=[cur_dt], ts=[cur_ts], sc=[cur_sc])
        print(f"[{cur_idx}] Time: {cur_dt} | Ts: {cur_ts} | Score: {cur_sc}")
        source.stream(new_data, rollover=15 * 24 * 7)


# -------------------------------------------------------------------------
# STREAMING LOOP

cur_idx = 0
init_idx = initialize()
raw_ts, raw_dt = read_tsdt()
# print("check1")
# print(raw_ts[:1000])

# Create Data Source
source = ColumnDataSource(dict(dt=[], ts=[], sc=[]))

# Draw a graph
fig = figure(x_axis_type="datetime", x_axis_label="Datetime", plot_width=950, plot_height=650)
fig.title.text = "Realtime monitoring"
fig.line(source=source, x='dt', y='ts', line_width=1, alpha=.85, color='blue', legend='Observed data')
# fig.circle(source=source, x='dt', y='ts', line_width=2, line_color='blue', color='blue')
fig.line(source=source, x='dt', y='sc', line_width=2, alpha=.85, color='red', legend='Change-point score')
fig.legend.location = "top_left"

# Configuration of the axis
time_format = "%Y-%m-%d-%H-%M-%S"
fig.xaxis.formatter = DatetimeTickFormatter(
    seconds = [time_format],
    minsec  = [time_format],
    minutes = [time_format],
    hourmin = [time_format],
    hours   = [time_format],
    days    = [time_format],
    months  = [time_format],
    years   = [time_format]
)

fig.xaxis.major_label_orientation=radians(90)

# Configuration of the callback
curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, 500) #ms
