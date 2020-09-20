import copy
import sys
import time
import json
import logging, logging.config
import coloredlogs
from configparser import ConfigParser
from src.models.DAGMM import Meta_DAGMM
from src.models.LSTM_DAGMM import Meta_LSTM_DAGMM
sys.path.append('/home/esowc22/')
from utils import *

# =============================================================================
# Define Time in minutes
RETRAIN_TIME = 12     # Retrain once a week.
PREDICT_TIME = 4      # Predict once ever 4 minutes.
UPDATE_TIME = 4 

# =============================================================================
def initialize():
    model_set['DAGMM']['model'] = Meta_DAGMM()
    model_set['DAGMM']['test_end'] = model_set['DAGMM']['train_end'] + \
        model_set['DAGMM']['forecast_length'] + model_set['DAGMM']['skip_length']
    
    model_set['LSTM_DAGMM']['model'] = Meta_LSTM_DAGMM()
    # model_set['LSTM_AD']['model'] = LSTM_AD()
    # model_set['LSTM_ED']['model'] = LSTM_ED()
    # model_set['DONUT']['model'] = DONUT()
    # model_set['REBM']['model'] = REBM()
    logger.info("All Models Initialized.")

def update_model_set(model_set):
    model_set['DAGMM']['train_end'] += model_set['DAGMM']['forecast_length']
    model_set['DAGMM']['test_end'] += model_set['DAGMM']['forecast_length']

def update_progress(model_set):    
    new_model_set = copy.deepcopy(model_set)
    for model in new_model_set:
        new_model_set[model].pop('model', None)

    logger.info("Saving information to disk.")
    with open('current_model_set.json', 'w') as f:
        json.dump(new_model_set, f, indent=4)

def start_retrain(model_set, global_config):
    logger.info("Retraining all Models.")
    for model in model_set:
        model_dict = model_set[model]
        if(global_config['some_option'] and model_dict['toTrain']):
            model_dict['model'].train(model_dict)


def start_getpredict(model_set, global_config):
    logger.info("Evaluating all Models.")
    for model in model_set:
        model_dict = model_set[model]
        if(global_config['another_option'] and model_dict['toPredict']):
            model_dict['model'].predict(model_dict)

# =============================================================================
if __name__ == '__main__':
    # print("=" * get_terminal_width())
    print(" ANOMALY ".center(get_terminal_width(), '='))

    # Get Global config file
    config_object = ConfigParser()
    config_object.read("config.ini")
    global_config = config_object["TESTING"]

    # Set up logging
    logging.config.fileConfig(fname='log_config.ini')
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger,fmt="[%(asctime)s][%(name)s] %(message)s",)

    # Load Model Set
    with open('model_set.json') as f:
        model_set = json.load(f)
    logger.info(f"Loaded Model Set.")

    initialize()
    cur_time = 0
    
    while(True):
        if(cur_time % UPDATE_TIME == 0):
            update_model_set(model_set)
        
        if(cur_time % RETRAIN_TIME == 0):
            start_retrain(model_set, global_config)

        if(cur_time % PREDICT_TIME == 0):
            start_getpredict(model_set, global_config)
            update_progress(model_set)      

        time.sleep(1)   # Sleep for 1 seconds
        cur_time += 1

    print("=" * get_terminal_width())