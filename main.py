import copy
import sys
import time
import json
import threading
import logging, logging.config
import coloredlogs
from configparser import ConfigParser
from src.models.DAGMM import Meta_DAGMM
from src.models.LSTM_DAGMM import Meta_LSTM_DAGMM
from src.models.REBM import Meta_REBM
from src.models.LSTMED import Meta_LSTMED
from src.models.LSTMAD import Meta_LSTMAD
from src.models.DONUT import Meta_DONUT
from src.threads.train import TrainThread
from src.threads.predict import PredictThread
# from src.models.ForecastX import Meta_ForecastX
from utils import *


# =============================================================================

# def process_config(gconfig):
#     int_fields = ['RETRAIN_TIME', 'PREDICT_TIME', 'UPDATE_TIME']
#     list_fields = ['TRAIN_MODELS', 'PRED_MODELS']

#     # gconfig['TRAIN_MODELS'] = 10
#     print(type(gconfig))
#     gconfig = dict(gconfig)
#     print(type(gconfig))
#     exit(0)

#     for field in int_fields:
#         print(gconfig[field])
#         print(field)
#         print(type(field))
#         t = eval(gconfig[field])
#     for field in list_fields:
#         t = eval(gconfig[field])

def initialize():    
    model_set['DAGMM']['model'] = Meta_DAGMM()
    model_set['LSTM_DAGMM']['model'] = Meta_LSTM_DAGMM()
    model_set['LSTMED']['model'] = Meta_LSTMED()
    model_set['LSTMAD']['model'] = Meta_LSTMAD()
    model_set['REBM']['model'] = Meta_REBM()
    model_set['DONUT']['model'] = Meta_DONUT()
    try:
        model_set['ForecastX']['model'] = Meta_ForecastX()
    except:
        logger.info("Ignoring Forecast X")

    for model in model_set:       
        if(model in gconfig['train_models']):
            m_dict = model_set[model]
            weight_folder = "/".join(m_dict["weight_path"].split("/")[:-1]) 
            os.makedirs(weight_folder, exist_ok=True)

        if(model in gconfig['pred_models']):
            m_dict = model_set[model]
            prediction_folder = "/".join(m_dict["predictions_path"].split("/")[:-1]) 
            os.makedirs(prediction_folder, exist_ok=True)
            m_dict['test_end'] = m_dict['train_end'] + m_dict['forecast_length'] + m_dict['skip_length']
            open(m_dict['predictions_path'], 'w').close()      # Clean existing predictions

    logger.info("All Models Initialized.")

def update_model_set(model_set):
    for model in model_set:
        if(model in gconfig['pred_models']):
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

def start_retrain(model_set, gconfig):
    # logger.info("Retraining all Models.")
    # for model in model_set:
    #     if(model in gconfig['train_models']): 
    #         m_dict = model_set[model]
    #         check = m_dict['model'].train(m_dict)
    #         if(check != "OK"): kill_me(check)

    train_thread = TrainThread("thread0", logger)
    train_thread.run(model_set, gconfig)
    print("Done Training")

# TODO: Return "OK" from all methods when finish training/preds.
def start_getpredict(model_set, gconfig):
    # logger.info("Evaluating all Models.")
    # for model in model_set:
    #     if(model in gconfig['pred_models']):
    #         m_dict = model_set[model]
    #         check = m_dict['model'].predict(m_dict)
    #         if(check != "OK"): kill_me(check) 

    predict_thread = PredictThread("thread1", logger)
    predict_thread.run(model_set, gconfig)
    print("Done Evaluations")
    

def kill_me(reason):
   """ Called when the program needs to exit. """
   logger.info(f"Exiting, reason: {reason}")
   exit(0)

# =============================================================================


if __name__ == '__main__':  
    print(" ~ BlackSwan ~ ".center(get_terminal_width(), '='))

    # Get Global config file
    config_object = ConfigParser()
    config_object.read("config.ini")
    gconfig = config_object["DEFAULT"]
    gconfig = dict(gconfig)
    for field in gconfig.keys():
        gconfig[field] = eval(gconfig[field])

    # Set up logging
    logging.config.fileConfig(fname='log_config.ini')
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', 
                        logger=logger,
                        fmt="[%(asctime)s][%(name)s] %(message)s")

    # Load Model Set
    with open('model_set.json') as f:
        model_set = json.load(f)
    logger.info(f"Loaded Model Set.")    # print("=" * get_terminal_width())


    initialize()
    cur_time = 0
    
    # train_thread = TrainThread(1)
    # predict_thread = PredictThread(2)

    # TODO: Check for end of input, and fix it.
    while(True):    
        logger.info("Time: " + str(cur_time))

        if(cur_time % int(gconfig['update_time']) == 0):
            update_model_set(model_set)
        
        if(cur_time % int(gconfig['retrain_time']) == 0):   
            # train_thread.start(model_set, gconfig)
            start_retrain(model_set, gconfig)

        if(cur_time % int(gconfig['predict_time']) == 0):
            # predict_thread.start(model_set, gconfig)
            start_getpredict(model_set, gconfig)
            # update_progress(model_set)      

        time.sleep(1)   # Sleep for 1 seconds
        cur_time += 1

    print("=" * get_terminal_width())