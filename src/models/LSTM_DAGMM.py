# Use: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/esowc22/anaconda3/lib/

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
sys.path.append('/home/esowc22/')
from utils import *

# ==================================================================================================

class Meta_LSTM_DAGMM():
    def __init__(self):
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
        self.test_begin = None
        self.test_end = None
        self.logger.info("Initialization Succesful.")        

    def get_update(self, config):
        os.makedirs(config['weight_path'], exist_ok=True)
        os.makedirs(config['predictions_path'], exist_ok=True)
        self.ts_path = config['data_path'] + "/ts_bytes_database=marsod_240s.txt"
        self.dt_path = config['data_path'] + "/dt.txt"
        self.weight_path = config['weight_path'] + '/run1.pth'
        self.predictions_path = config['predictions_path'] + '/run1.pth'
        self.train_begin = config['train_begin']
        self.test_begin = config['test_begin']
        self.test_end = config['test_end']
        self.ts = pd.DataFrame(np.loadtxt(self.ts_path))
        self.dt = load_dt(self.dt_path)

    def train(self, config):
        '''Train and save the weights'''
        self.logger.info("Starting Training.")

        self.get_update(config)
        train_ts = self.ts[self.train_begin:self.test_begin]
        train_dt = self.dt[self.train_begin:self.test_begin]

        model = DAGMM(autoencoder_type = LSTMEDModule,
                lambda_energy = 0.2, lr=1e-3, batch_size=50, gmm_k=3, num_epochs=1, sequence_length=20)   
        model.fit(train_ts)
        torch.save(model, self.weight_path)
        self.logger.info(f"Training Done. Saved weights at: {self.weight_path}")

    def predict(self, config):
        '''Predict and save the predictions'''
        self.logger.info("Starting Evaluation.")

        self.get_update(config)
        test_ts = self.ts[self.test_begin:self.test_end]
        test_dt = self.dt[self.test_begin:self.test_end]

        model = DAGMM(lambda_energy = 0.2, lr=1e-3, batch_size=50, gmm_k=3, num_epochs=1, sequence_length=20)
        model = torch.load(self.weight_path)
        predictions = model.predict(test_ts)
        torch.save(predictions, self.predictions_path)
        self.logger.info(f"Evaluation Done. Saved predictions at: {self.predictions_path}")
    

    







