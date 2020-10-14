# -------------------------------------------------------------------------
# IMPORTS

import json
import warnings
import numpy as np
import pandas as pd
from math import radians
from datetime import datetime, timedelta
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Span
from bokeh.plotting import figure
from bokeh.layouts import gridplot
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# FUNCTIONS

def initialize():   
    with open('model_set.json') as f:
        model_set = json.load(f)
    return model_set['DAGMM']['train_end']

def read_tsdt():
    ROOT_PATH = "data/June30_mars_4min_an/"
    file = 'ts_bytes_database=marsod_240s.txt'
    raw_ts = pd.read_csv(ROOT_PATH + file, header=None).values.reshape(-1, )
    # raw_ts = raw_ts[:] / (np.mean(raw_ts) * 10)
    time_strings = pd.read_csv(ROOT_PATH + "dt.txt", header = None).values[:,0]
    raw_dt = [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in time_strings]
    return raw_ts, raw_dt

def read_preds():
    # TODO: Edit PREDS_PATH to a variable.    
    PREDS_PATH = "predictions/ForecastX/sc.txt"
    # PREDS_PATH = "predictions/LSTM_DAGMM/sc_window=1000_forecast=1.txt"

    try:
        preds = pd.read_csv(PREDS_PATH, header=None).values.reshape(-1, )
        preds = preds[:] / (np.mean(preds) * 10)
    except:
        preds = []
    return preds

def update_data():
    global raw_ts
    global raw_dt
    # global init_idx
    global cur_idx
    global all_scores
    
    preds = read_preds()
    preds - np.array(preds)
    cur = 0

    while(cur_idx < len(preds) and cur < max_points):
        cur_ts = raw_ts[cur_idx + init_idx]
        cur_dt = raw_dt[cur_idx + init_idx]
        cur_sc = preds[cur_idx]                       # * 1e12
        all_scores.append(cur_sc)
        cur_idx += 1
        cur += 1
   
        print(f"[{cur_idx}] Time: {cur_dt} | Ts: {cur_ts} | Score: {cur_sc}")
        source11.stream(dict(dt=[cur_dt], ts=[cur_ts]), rollover=rollover)
        source21.stream(dict(dt=[cur_dt], sc=[cur_sc]), rollover=rollover)

def make_figure(fig_title = "RealTime monitoring"):
    ffig = figure(x_axis_type="datetime",plot_width=pwidth, plot_height=pheight)
    ffig.legend.location = "top_right"
    ffig.xaxis.formatter = DatetimeTickFormatter(seconds=[tf], minsec=[tf], minutes=[tf], hourmin=[tf], hours=[tf], days=[tf], months=[tf], years=[tf])
    # ffig.title.text = fig_title
    # ffig.xaxis.major_label_orientation=radians(15)
    return ffig

# ================================================================================================
# STREAMING LOOP

### PARAMS ###
rollover = 2 * (15*24)           # Rollover length
max_points = 1e9                 # Max points to plot at a time.
refresh_interval = 50         # Refresh inteval in ms.
pheight = 400
pwidth = 1400
tf = "%Y/%b/%d-%H-%M-%S"       

cur_idx = 0                     # Index of the plotted point.
init_idx = initialize()         # Index in the original TS from where evaluation beings.
raw_ts, raw_dt = read_tsdt()    # Read in the raw Time Series from the `data` folder.
all_scores = []

# Figure [1,1]
fig11 = make_figure("Time Series")
source11 = ColumnDataSource(dict(dt=[], ts=[]))
fig11.line(source=source11, x='dt', y='ts', line_width=2, alpha=.7, color='green', legend='Time Series')

# Figure [2,1]
fig21 = make_figure("Anomaly Score")
source21 = ColumnDataSource(dict(dt=[], sc=[]))
fig21.line(source=source21, x='dt', y='sc', line_width=2, alpha=.7, color='red', legend='Aomaly Score')

# Combine all Figures
fig = gridplot([[fig11],[fig21]])

# Configuration of the callback
curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, refresh_interval)
