# ---------------------------------------------------------------------
### IMPORTS ###
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cm as cm
from tqdm.auto import trange, tqdm
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings('ignore')
 
# ---------------------------------------------------------------------
### DEFAULTS ###
LOG_DIR = "/home/esowc22/ESoWC_backup/Project-22/Generate_data"
LABEL_DIR = "/home/esowc22/ESoWC_backup/Project-22/Generate_metadata"
TS_OUTPUT_DIR = "/home/esowc22/ESoWC_backup/Project-22/Generate_outputs/Sample2"
GRAPH_OUTPUT_DIR = "/home/esowc22/ESoWC_backup/Project-22/Generate_outputs/Sample_Graphs2"
startTimeSeries = datetime(2020, 5, 1, 0, 0, 0)
endTimeSeries = datetime(2020, 5, 2, 23, 59, 59)

# ---------------------------------------------------------------------
### LOAD THE DATA ###

# Load Log Paths : May need custom logic based on input files.
logPaths = ["may_{:02d}.csv".format(i) for i in range(1,3)]
df = pd.read_csv(os.path.join(LOG_DIR, logPaths[0]))
for i in trange(1,len(logPaths)):
    logPathFull = os.path.join(LOG_DIR, logPaths[i])
    df_day = pd.read_csv(logPathFull)
    df = pd.concat([df,df_day])

# Filter global dataset
df = df[df.status == 'ok']
df = df[df.verb == 'retrieve']
df.fields = df.fields.fillna(0)

# Add request column to deal with feature
df['requests'] = np.array(['1' for i in range(len(df))])  
df = df.drop(['status', 'retTimes', 'transfertime'], axis=1)

# PreCompute start/stop times for efficiency
df_startDateTimes = []
for val in tqdm(df.startTimes.values):
    df_startDateTimes.append(datetime.strptime(val, "%Y-%m-%d %H:%M:%S"))
df_startDateTimes = np.array(df_startDateTimes)

df_stopDateTimes = []
for val in tqdm(df.stopTimes.values):
    df_stopDateTimes.append(datetime.strptime(val, "%Y-%m-%d %H:%M:%S"))
df_stopDateTimes = np.array(df_stopDateTimes)

# Reduce DF size for efficiency
df = df.drop(['startTimes', 'stopTimes', 'verb',], axis=1, errors='ignore')
df['elapsed'] = df['elapsed'].values.astype(np.int32)
df['fields'] = df['fields'].values.astype(np.int32)
df['requests'] = df['requests'].values.astype(np.int8)

# print(len(df), len(df_startDateTimes), len(df_startDateTimes))

# ---------------------------------------------------------------------
### LOAD THE DOWNTIMES ###

def StrToDateTime1(string):
    '''
    Converts string denoting date-time to datetime object.
    Sample Input: '20200430-084202'
    '''

    dateStr = string.split('-')[0]
    timeStr = string.split('-')[1]
    year, month, day = int(dateStr[0:4]), int(dateStr[4:6]), int(dateStr[6:8])
    hours, minutes, seconds = int(timeStr[0:2]), int(timeStr[2:4]), int(timeStr[4:6])
    return datetime(year, month, day, hours, minutes, seconds)

endDate = datetime(2020,7,1,0,0,0)
downTimeFiles = os.listdir(LABEL_DIR)
downTimes = {}

for serverFile in downTimeFiles:
    serverDownDataPath = os.path.join(LABEL_DIR, serverFile)
    downServer = serverFile.split(".")[0].split('_')[0]
    serverData = np.loadtxt(serverDownDataPath, dtype=str)[:,0]
    if(downServer in downTimes.keys()):
        downTimes[downServer] = np.concatenate([downTimes[downServer],serverData])
    else:
        downTimes[downServer] = serverData
    
for key in downTimes.keys():
    downTimes[key].sort()
    downTimes[key] = [StrToDateTime1(t) for t in downTimes[key]]
    downTimes[key] = [t for t in downTimes[key] if (t < endDate) ]

numAnomalies = 0
for key in downTimes.keys():
    numAnomalies += len(downTimes[key])

# ---------------------------------------------------------------------
### GENERATE THE TIME-SERIES ###

def makeTS(
    field,
    filterList,
    binInterval,
    toAverageLoad = True
    ):
    '''
    Returns a time series according to paramter specification.

    filterList : List of filters to apply to the timeSeries.
    binInterval : Number of seconds to bin into a single data point.
    toAverageLoad : Whether to average the fieldData across minutes.
    '''
    
    # Get data df
    data = df.copy() #df.copy()
    startDateTimes = df_startDateTimes.copy()
    stopDateTimes = df_stopDateTimes.copy()
    
    # Filter rows according to filterList columns
    for i in range(int(len(filterList)/2)):
        filterBy = filterList[2*i]
        filterVal = filterList[2*i+1]
        mask1 = (data[filterBy] == filterVal)
        data = data[mask1]
        startDateTimes = startDateTimes[mask1]
        stopDateTimes = stopDateTimes[mask1]

    # Remove rows which dont have proper field value 
    mask2 = data[field].notna()
    data = data[mask2]
    startDateTimes = startDateTimes[mask2]
    stopDateTimes = stopDateTimes[mask2]

    # Get Field data
    fieldData = data[field].values
    fieldData = np.array(fieldData, dtype=np.float32)
    if(len(fieldData) <= 100):
        print("[!] No data found matching filters. Exiting.")
        return None, None
    
    # Get Elapsed-time data
    isNaN = data.elapsed.isna().any()
    assert not(isNaN), "NaNs in series."
    elapsedSecs = data.elapsed.values
    elapsedSecs = np.array(elapsedSecs)
    
    startTimeSeries = datetime(2020, 4, 27, 0, 0, 0)
    endTimeSeries = datetime(2020, 6, 30, 23, 59, 59)
    
    timeDelta = timedelta(seconds=binInterval)       # Time Difference when binning data points.
    totalPoints = math.ceil((endTimeSeries - startTimeSeries) / timeDelta)
    totalDataForInterval = np.zeros([totalPoints])   # Stores final field values
    totalPoints

    for i in trange(len(data), leave=False):         # Loop over all logs
        endTime = stopDateTimes[i]          # Log end time
        startTime = startDateTimes[i]       # Log start time
        rawLoad = fieldData[i]              # Raw Load for the log
        if(toAverageLoad and field != 'requests'):
            secs = max(int(elapsedSecs[i]), 1)
            timePeriods = math.ceil(timedelta(seconds=secs) / timeDelta)
            load = rawLoad / timePeriods           # Normalized load for the log
        else:
            load = rawLoad

        # Loop over all the time Periods for when log is active
        startIdx = max(math.floor((startTime - startTimeSeries) / timeDelta),0)
        endIdx = math.floor((endTime - startTimeSeries) / timeDelta)
        
        if(endIdx < totalPoints): 
            totalDataForInterval[endIdx] += load
                
    timePoints = [startTimeSeries + (idx * timeDelta) for idx in range(totalPoints)]
    timePoints = np.array(timePoints)          
    return totalDataForInterval, timePoints


os.makedirs(TS_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
    
def SaveTs(data, fileName):
    '''Save Time Series to Disk'''
    path = os.path.join(TS_OUTPUT_DIR, fileName + ".txt")
    np.savetxt(path, data)
    
def SaveDt(data, fileName):
    '''Save Timestamps to Disk'''
    path = os.path.join(TS_OUTPUT_DIR, fileName + ".txt")
    np.savetxt(path, data, fmt='%s')

def Plot(dataTS, timePoints, yLabel, figTitle="", toSave=False):
    col = ['red', 'blue', 'green', 'brown', 'yellow', 'purple'] 
    col = [cm.Dark2(i) for i in range(len(downTimes.keys()))]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in col]
    
    plt.figure(figsize=[16,7])
    plt.legend(lines, downTimes.keys())           
    s = 0/14
    e = 14.0/14
    st = int(len(timePoints) * s)
    end = int(len(timePoints) * e)-1
    plt.plot(timePoints[st:end], dataTS[st:end], 'k-', linewidth=0.7)
    plt.grid()
    plt.title(figTitle)
    plt.xlabel("Time")
    plt.ylabel(yLabel)
    
    ax = plt.axes()
    locator = mdates.AutoDateLocator(minticks=20, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
        
    for idx,key in enumerate(downTimes.keys()):
        for point in downTimes[key]:
            if(point < timePoints[end] and point > timePoints[st]):
                point = point - timedelta(minutes=0)
                plt.axvline(point, color=col[idx], zorder=1)
    
    if(toSave):
        path = os.path.join(GRAPH_OUTPUT_DIR, figTitle + ".png")
        plt.savefig(path, dpi=200)
    
    plt.show()


fieldMappings = {
    'requests' : "Requests",
    'bytes' :    "Retrieved Volume",
    'written' :  "Delivered Volume",
    'elapsed' :  "Elapsed Time",
    'fields' :   "Number of Completed Requests"
}

binInterval = 60*4
fieldColumns = ["requests", "written", "bytes", "elapsed", "fields"]
filterColumns = ["database", "class", "stream", "type"]

dt_saved = False

for fieldCol in fieldColumns:
    for filterCol in filterColumns:
    
        if(filterCol == "database"):
            filterVals = list(set(df[filterCol].value_counts().index[:0]) | set(downTimes.keys()))
        else:
            filterVals = list(df[filterCol].value_counts().index[:5])
        
        for filterVal in list(filterVals):
            filename = fieldCol + "_" + filterCol + "=" + filterVal  \
                        + "_" + str(binInterval) + "s"
            
            description = fieldMappings[fieldCol] 
            print("Saving:", filename)

            # Use makeTS(fieldCol, filterCol, filterVal) to generate the TS
            ts, times = makeTS(fieldCol, [filterCol, filterVal], binInterval=binInterval)
            
            # Save all the Time Series.
            if(ts is not None and times is not None):
                # Plot(ts, times, description, filename, toSave=True)  # Plot the TS Graph?  
                SaveTs(ts, filename) 
                if not dt_saved:
                    SaveDt(times, filename)          
                    dt_saved = True