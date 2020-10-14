# NOTE: Use export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/esowc22/anaconda3/lib/
import os
import sys
#TODO - see if there are better ways to handle this than adding a path
print(os.getcwd())
sys.path.append("src")
import json
import logging, logging.config
import coloredlogs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from utils import *
from forecast_x import forecast_x
from forecast_utils import get_complete_ts_anomaly_score
# ==================================================================================================

TS_NAME = "/ts_bytes_database=marsod_240s.txt"
DT_NAME = "/dt.txt"

class Meta_ForecastX():
    def __init__(self):
        '''Initialize the class variables'''
        logging.config.fileConfig(fname='log_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG', logger=self.logger,fmt="[%(asctime)s][%(name)s] %(message)s",)
        self.ts = None
        self.dt = None
        self.ts_path = None
        self.dt_path = None
        self.predictions_path = None
        self.train_begin = None
        self.train_end = None
        self.test_begin = None
        self.test_end = None
        self.input_window_length = None
        self.forecast_length = None
        self.frequency = None
        self.skip_length = None
        self.logger.info("Initialization Succesful.")        

    def get_update(self, config):
        '''Reload params from updated config'''
        
        self.input_window_length = config['input_window_length']
        self.forecast_length = config['forecast_length']
        self.frequency = config['frequency']
        self.skip_length = config['skip_length']
        self.ts_path = config['data_path'] + TS_NAME
        self.dt_path = config['data_path'] + DT_NAME
        self.predictions_path = config['predictions_path']
        self.train_begin = config['train_begin']
        self.train_end = config['train_end']
        self.test_begin = config['test_end'] - self.input_window_length - self.forecast_length
        self.test_end = config['test_end']
        self.ts = pd.DataFrame(np.loadtxt(self.ts_path)).values
        self.ts = np.squeeze(self.ts).tolist()
        self.dt = load_dt(self.dt_path)
        
    def train(self, config):
        self.logger.info(f"ForecastX train called - This method does not require training. Continuing")
        return "OK"        

    def predict(self, config):
        '''Predict and save the predictions'''  

        self.get_update(config)
        test_ts = self.ts[self.test_begin:self.test_end]
        test_dt = self.dt[self.test_begin:self.test_end]
        if(len(test_ts) == 0): 
            return "End of Test input."

        input_window = test_ts[-(self.input_window_length + self.forecast_length):-self.forecast_length]
        ground_truth_window = test_ts[-self.forecast_length:]
        self.logger.info(f"Starting Evaluation. Points: [{self.test_begin}, {self.test_end}]")
        forecast_x_obj = forecast_x.forecast(input_window, self.frequency, self.forecast_length)
        model = forecast_x_obj.best_model()
        forecast = forecast_x_obj.get_forecast(model)
        
        predictions = get_complete_ts_anomaly_score(ground_truth_values = ground_truth_window, 
                                                      predicted_values = forecast, 
                                                      epsilon = 100, 
                                                      quantile_scaling = False,
                                                      threshold = -1000000)

        with open(self.predictions_path, "a") as myfile:
            for i in range(-self.forecast_length-self.skip_length, -self.skip_length):
                myfile.write(str(predictions[i]) + '\n')
        
        self.logger.info(f"Evaluation Done. Saved predictions at: {self.predictions_path}")
        return "OK"
