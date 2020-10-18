# -------------------------------------------------------------------------
# IMPORTS

import json
import warnings
import time
import numpy as np
import pandas as pd
from math import radians
from datetime import datetime, timedelta
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Span, Range1d
from bokeh.models import Label

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
    PREDS_PATH = "predictions/LSTM_DAGMM/sc_final.txt"

    try:
        preds = pd.read_csv(PREDS_PATH, header=None).values.reshape(-1, )
        preds = preds[:] / (np.mean(preds) * 10)
    except:
        preds = []


    preds = np.array(preds)
    assert np.all(preds > 0)
    preds = np.log(preds + 2)
    return preds

def update_data():
    global raw_ts
    global raw_dt
    # global init_idx
    global cur_idx  
    global all_scores
    global max_score
    global min_score
    # global fig21
    
    preds = read_preds()
    preds = np.array(preds)
    
    cur = 0

    while(cur_idx < len(preds) and cur < max_points):
        cur_ts = raw_ts[cur_idx + init_idx]
        cur_dt = raw_dt[cur_idx + init_idx]
        cur_sc = 3 - preds[cur_idx]                      
        all_scores.append(cur_sc)
        cur_idx += 1
        cur += 1

        if(max_score != None): max_score = max(max_score, cur_sc)
        else: max_score = cur_sc
        if(min_score != None): min_score = min(min_score, cur_sc)
        else: min_score = cur_sc

        high_line.location = max_score
        low_line.location = min_score
        fig21.title.text = "Anomaly Scores : [%1.4f, %1.4f]"%(min_score, max_score)

        print(f"[{cur_idx}] Time: {cur_dt} | Ts: {cur_ts} | Score: {cur_sc}")
        source11.stream(dict(dt=[cur_dt], ts=[cur_ts]), rollover=rollover)
        source21.stream(dict(dt=[cur_dt], sc=[cur_sc]), rollover=rollover)
        print(high_line.location, low_line.location)
        print(fig21.y_range.start, fig21.y_range.end)


def make_figure(fig_title = "RealTime monitoring", y_type = "linear"):
    ffig = figure(x_axis_type="datetime", 
                y_axis_type=y_type, 
                plot_width=pwidth, 
                plot_height=pheight)
    
    
    ffig.legend.location = "top_right"
    ffig.xaxis.formatter = DatetimeTickFormatter(seconds=[tf], minsec=[tf], minutes=[tf], hourmin=[tf], hours=[tf], days=[tf], months=[tf], years=[tf])
    return ffig

# ================================================================================================
# STREAMING LOOP

### PARAMS ###
rollover = 2 * (15*24)           # Rollover length
max_points = 50                  # Max points to plot at a time.
refresh_interval = 1000          # Refresh inteval in ms.
pheight = 400
pwidth = 1400
tf = "%Y/%b/%d-%H-%M-%S"       

cur_idx = 0                     # Index of the plotted point.
init_idx = initialize()         # Index in the original TS from where evaluation beings.
raw_ts, raw_dt = read_tsdt()    # Read in the raw Time Series from the `data` folder.
all_scores = []
max_score = None
min_score = None

# Figure [1,1]
fig11 = make_figure("Time Series", y_type='linear')
source11 = ColumnDataSource(dict(dt=[], ts=[]))
fig11.line(source=source11, x='dt', y='ts', line_width=2, alpha=.7, color='green', legend='Time Series')
fig11.title.text = "Marsod Bytes [Aggregated over 4 minute intervals]"


# Figure [2,1]
fig21 = make_figure("Anomaly Score", y_type='log')
source21 = ColumnDataSource(dict(dt=[], sc=[]))
fig21.line(source=source21, x='dt', y='sc', color='red', line_width=2, alpha=.7, legend='Anomaly Score')

high_line = Span(location=None, dimension='width', line_color='black', line_width=1, line_dash='dashed')
low_line = Span(location=None, dimension='width', line_color='black', line_width=1, line_dash='dashed')
fig21.renderers.extend([high_line, low_line])
fig21.title.text = "Anomaly Score Range : []"
fig21.yaxis.axis_label = "Anomaly Scores [ Logarithmic ]"

# Combine all Figures
fig = gridplot([[fig11],[fig21]])

# Configuration of the callback
curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, refresh_interval)
