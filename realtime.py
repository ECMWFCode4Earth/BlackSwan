# -------------------------------------------------------------------------
# IMPORTS

import os
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

# %load_ext autoreload
# %autoreload 2     

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure
from math import radians
from pytz import timezone

# -------------------------------------------------------------------------
# LOAD DATA

ROOT_PATH = "data/June30_mars_4min/"

file = 'ts_bytes_database=marsod_240s.txt'
raw_ts = pd.read_csv(ROOT_PATH + file, header=None).values.reshape(-1, )
raw_ts = raw_ts[:] / (np.mean(raw_ts) * 10)

time_strings = pd.read_csv(ROOT_PATH + "dt.txt", header = None).values[:,0]
raw_dt = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in time_strings]

# -------------------------------------------------------------------------
# STREAMING LOOP

scores = []
ts = []
dt = []

def get_new_data():
    global ts
    global dt
    global raw_ts
    global raw_dt
    ts.append(raw_ts[0])
    dt.append(raw_dt[0])
    raw_ts = np.delete(raw_ts, 0)
    raw_dt = np.delete(raw_dt, 0)

def update_data():
    global scores
    global dt
    global dt
    get_new_data()
    score = ts[-1] * 1.3
    scores.append(score)
    new_data = dict(dt=[dt[-1]], ts=[ts[-1]], sc=[scores[-1]])
    print(score)
    source.stream(new_data, rollover=500)

# Create Data Source
source = ColumnDataSource(dict(dt=[], ts=[], sc=[]))

# Draw a graph
fig = figure(x_axis_type="datetime", x_axis_label="Datetime", plot_width=950, plot_height=650)
fig.title.text = "Realtime monitoring with Banpei"
fig.line(source=source, x='dt', y='ts', line_width=1, alpha=.85, color='blue', legend='Observed data')
# fig.circle(source=source, x='dt', y='ts', line_width=2, line_color='blue', color='blue')
fig.line(source=source, x='dt', y='sc', line_width=2, alpha=.85, color='red', legend='Change-point score')
fig.legend.location = "top_left"

# Configuration of the axis
time_format = "%Y-%m-%d-%H-%M-%S"
fig.xaxis.formatter = DatetimeTickFormatter(
    seconds=[time_format],
    minsec =[time_format],
    minutes=[time_format],
    hourmin=[time_format],
    hours  =[time_format],
    days   =[time_format],
    months =[time_format],
    years  =[time_format]
)

fig.xaxis.major_label_orientation=radians(90)

# Configuration of the callback
curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, 5) #ms
