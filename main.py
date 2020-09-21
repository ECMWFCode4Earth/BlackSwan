import copy
import sys
import time
import json
import logging, logging.config
import coloredlogs
from configparser import ConfigParser
from src.models.DAGMM import Meta_DAGMM
from src.models.LSTM_DAGMM import Meta_LSTM_DAGMM
from src.models.REBM import Meta_REBM
from src.models.LSTMED import Meta_LSTMED
from src.models.LSTMAD import Meta_LSTMAD
from src.models.DONUT import Meta_DONUT
from utils import *

# =============================================================================
def initialize():
    model_set['DAGMM']['model'] = Meta_DAGMM()
    model_set['LSTM_DAGMM']['model'] = Meta_LSTM_DAGMM()
    model_set['LSTMED']['model'] = Meta_LSTMED()
    model_set['LSTMAD']['model'] = Meta_LSTMAD()
    model_set['REBM']['model'] = Meta_REBM()
    model_set['DONUT']['model'] = Meta_DONUT()

    for model in model_set:
        m_dict = model_set[model]
        weight_folder = "/".join(m_dict["weight_path"].split("/")[:-1]) 
        prediction_folder = "/".join(m_dict["predictions_path"].split("/")[:-1]) 
        os.makedirs(weight_folder, exist_ok=True)
        os.makedirs(prediction_folder, exist_ok=True)
        m_dict['test_end'] = m_dict['train_end'] + m_dict['forecast_length'] + m_dict['skip_length']
        open(m_dict['predictions_path'], 'w').close()

    logger.info("All Models Initialized.")

def update_model_set(model_set):
    for model in model_set:
        m_dict = model_set[model]
        m_dict['train_end'] += m_dict['forecast_length']
        m_dict['test_end'] += m_dict['forecast_length']

# def update_progress(model_set):    
#     '''
#     Save a copy of model_set as `current_model_set.json` so that plot.py 
#     can read off test begin, end indices from it.

#     NOTE: Not needed, currently plot.py just uses the length of the prediceted 
#     output file to get test indices.
#     '''
#     new_model_set = copy.deepcopy(model_set)
#     for model in new_model_set:
#         new_model_set[model].pop('model', None)

#     logger.info("Saving information to disk.")
#     with open('current_model_set.json', 'w') as f:
#         json.dump(new_model_set, f, indent=4)

def start_retrain(model_set, global_config):
    logger.info("Retraining all Models.")
    for model in model_set:
        m_dict = model_set[model]
        if(model in gconfig['TRAIN_MODELS']):  
            m_dict['model'].train(m_dict)


def start_getpredict(model_set, global_config):
    logger.info("Evaluating all Models.")
    for model in model_set:
        m_dict = model_set[model]
        if(model in gconfig['PRED_MODELS']):
            m_dict['model'].predict(m_dict)

# =============================================================================
if __name__ == '__main__':
    # print("=" * get_terminal_width())
    print(" ~ BlackSwan ~ ".center(get_terminal_width(), '='))

    # Get Global config file
    config_object = ConfigParser()
    config_object.read("config.ini")
    gconfig = config_object["DEFAULT"]

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
        if(cur_time % int(gconfig['UPDATE_TIME']) == 0):
            update_model_set(model_set)
        
        if(cur_time % int(gconfig['RETRAIN_TIME']) == 0):
            start_retrain(model_set, gconfig)

        if(cur_time % int(gconfig['PREDICT_TIME']) == 0):
            start_getpredict(model_set, gconfig)
            # update_progress(model_set)      

        time.sleep(1)   # Sleep for 1 seconds
        cur_time += 1

    print("=" * get_terminal_width())