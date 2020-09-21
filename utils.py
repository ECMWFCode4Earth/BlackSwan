import json
import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cm as cm
from tqdm.notebook import trange, tqdm
from datetime import datetime, timedelta 

def get_terminal_width():
    return int(os.popen('stty size', 'r').read().split()[1])

def resize_jupyter(width=80):
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:" + str(width) +"% !important; }</style>"))

def load_dt(path):
    with open(path) as file: 
        data = file.read() 
    dt = [datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") for ts_str in data.split('\n') if ts_str != '']
    return dt
    
def load_labels(path):
    with open(path, 'r') as openfile: 
        labels = json.load(openfile)

    for label_type in labels:
        for database_type in labels[label_type]:
            for interval_idx in range(len(labels[label_type][database_type])):
                interval = labels[label_type][database_type][interval_idx]
                start = datetime.strptime(interval[0], "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(interval[1], "%Y-%m-%d %H:%M:%S")
                labels[label_type][database_type][interval_idx] = [start, end]

    return labels
    
    

def plot_self(ts_list, times, down, st, end, labels, yLabel="", figTitle="", toSave=False, addDay=False, flag_thresh=None):
    col = ['deepskyblue', 'peachpuff', 'darkgrey', 'cornflowerblue']
    legendLabels = ['Flags', 'Self-Labels', 'Self-Labels [C]', 'DownTimes']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in col]
    
    if(addDay):
        curDay = times[int(st+(end-st)/2)]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        months = ['Dum', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
        curDay = times[int(st+(end-st)/2)]
        weekDay = days[curDay.weekday()]
        year = "2020"
        month = months[curDay.month]
        day = curDay.day
        prefixStr = " [{:02d} {}, {}] [{}]".format(day,month,year,weekDay)
        figTitle += prefixStr
    
    if(end-st <= 180): marker = '.'
    else: marker = ""
    
    plt.figure(figsize=[16,7])
    plt.legend(lines, legendLabels)
    
    ts_col_list = ['black', 'blueviolet', 'mediumseagreen', 'peru'] 
    for idx in range(len(ts_list)):
        plt.plot(times[st:end], ts_list[idx][st:end], linestyle='-', zorder=10, color=ts_col_list[idx], marker=marker)
    
    plt.grid()
    plt.title(figTitle)
    plt.xlabel("Time")
    plt.ylabel(yLabel)
    # plt.yticks(np.arange(0.0,7,0.5) * 1e11)
    # plt.xticks(np.arange(times[st],times[end], timedelta(minutes=60)))
    
    # Flags
    if(flag_thresh != None):
        plt.axhline(flag_thresh, linestyle='--', color='deepskyblue', zorder=1)
        flag_points = times[ts < flag_thresh]
        for point in flag_points:
            if(point < times[end-1] and point > times[st]):
                plt.axvline(point, color=col[0], zorder=1, ymin=0, ymax=1)
            
    # SelfLabels Standard
    selfLabels = labels['selfLabels']
    for db in selfLabels.keys():
        if(db in down):
            intervals = selfLabels[db]
            for interval in intervals:
                points = fill_dips(interval)
                for point in points:
                    if(point < times[end-1] and point > times[st]):
                        plt.axvline(point, color=col[1], zorder=1, ymin=0, ymax=0.95)
                        
    # SelfLabels Challenging
    selfLabels_challenge = labels['selfLabels_challenge']
    for db in selfLabels_challenge.keys():
        if(db in down):
            intervals = selfLabels_challenge[db]
            for interval in intervals:
                points = fill_dips(interval)
                for point in points:
                    if(point < times[end-1] and point > times[st]):
                        plt.axvline(point, color=col[2], zorder=1, ymin=0, ymax=0.95)
 
    # Clubbed Downtimes
    clubbedDownTimes = labels['clubbedDownTimes']
    for db in clubbedDownTimes.keys():
        if(db in down):
            intervals = clubbedDownTimes[db]
            for interval in intervals:
                points = fill_dips(interval)
                for point in points:
                    if(point < times[end-1] and point > times[st]):
                        plt.axvline(point, color=col[3], zorder=1, ymin=0.05, ymax=1)
 
    ax = plt.axes()
    ax2 = ax.twiny()
    ax2.set_xlim(0,1)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
 
    if(toSave):
        path = os.path.join(folderGraphs2, figTitle + ".png")
        plt.savefig(path, dpi=200)
    plt.show()
    
    
def fill_dips(dips):
    start, end = dips[0], dips[-1]    
    newDips = []
    while(start <= end):
        newDips.append(start)
        start = start + timedelta(seconds=30)
    return newDips


def get_stend(start=None, delay=None, anchor=None, delta=None, l=None):
    globalStart = datetime(2020,4,27,0,0,0)
    globalStop = datetime(2020,6,30,23,59,59)

    if(start != None and delay != None):
        curStart = start
        curStop = start + delay
    elif(anchor != None and delta != None):
        curStart = anchor - delta
        curStop = anchor + delta
    else:
        curStart = globalStart
        curStop = globalStop
    
    s = max((curStart - globalStart)/(globalStop - globalStart),0)
    e = min((curStop - globalStart)/(globalStop - globalStart),1)
    st = int(l*s)
    end = int(l*e)
    return st, end