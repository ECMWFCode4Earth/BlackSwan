import sys
import json
import os
import torch
import logging, logging.config
import coloredlogs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from lib.DeepADoTS.src.algorithms import DAGMM  
from lib.DeepADoTS.src.algorithms.lstm_enc_dec_axl import LSTMEDModule
from utils import *

# ==================================================================================================

TS_NAME = "/ts_bytes_database=marsod_240s.txt"
DT_NAME = "/dt.txt"

class Meta_LSTM_DAGMM():
    def __init__(self):
        '''Initialize the class variables'''
        logging.config.fileConfig(fname='log_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG', logger=self.logger,fmt="[%(asctime)s][%(name)s] %(message)s",)
        self.ts = None
        self.dt = None
        self.ts_path = None
        self.dt_path = None
        self.weight_path = None
        self.predictions_path = None
        self.train_begin = None
        self.train_end = None
        self.test_begin = None
        self.test_end = None
        self.forecast_length = None
        self.skip_length = None
        self.logger.info("Initialization Succesful.")        

    def get_update(self, config):
        '''Relaod params from updated config'''
        self.forecast_length = config['forecast_length']
        self.skip_length = config['skip_length']
        self.ts_path = config['data_path'] + TS_NAME
        self.dt_path = config['data_path'] + DT_NAME
        self.weight_path = config['weight_path']
        self.predictions_path = config['predictions_path']
        self.train_begin = config['train_begin']
        self.train_end = config['train_end']
        # self.test_begin = config['test_begin']
        self.test_begin = config['test_end'] - 1000
        self.test_end = config['test_end']
        self.ts = pd.DataFrame(np.loadtxt(self.ts_path))
        self.dt = load_dt(self.dt_path)

    def train(self, config):
        '''Train and save the weights'''

        self.get_update(config)
        train_ts = self.ts[self.train_begin:self.train_end]
        train_dt = self.dt[self.train_begin:self.train_end]
        self.logger.info(f"Starting Training. Points: [{self.train_begin}, {self.train_end}]")

        model = DAGMM(autoencoder_type = LSTMEDModule, **config['training_params'])   
        model.fit(train_ts)
        torch.save(model, self.weight_path)
        self.logger.info(f"Training Done. Saved weights at: {self.weight_path}")

    def predict(self, config):
        '''Predict and save the predictions'''  

        self.get_update(config)
        test_ts = self.ts[self.test_begin:self.test_end]
        test_dt = self.dt[self.test_begin:self.test_end]
        self.logger.info(f"Starting Evaluation. Points: [{self.test_begin}, {self.test_end}]")

        model = DAGMM(autoencoder_type = LSTMEDModule, **config['training_params'])
        model = torch.load(self.weight_path)
        predictions = model.predict(test_ts)

        fl = self.forecast_length
        with open(self.predictions_path, "a") as myfile:
            for i in range(-self.forecast_length-self.skip_length, -self.skip_length):
                myfile.write(str(predictions[i]) + '\n')
        
        self.logger.info(f"Evaluation Done. Saved predictions at: {self.predictions_path}")
