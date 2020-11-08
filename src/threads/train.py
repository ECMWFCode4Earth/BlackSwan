import time
import threading
import coloredlogs
import logging, logging.config

class TrainThread(threading.Thread):
    def __init__(self, threadID, logger):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = "TrainThread"
        self._is_running = True
        self.logger = logger

        # logging.config.fileConfig(fname='log_config.ini', 
        #                         disable_existing_loggers=False)
        
        # self.logger = logging.getLogger(__name__)
        # coloredlogs.install(level='DEBUG', 
        #                     logger=self.logger,
        #                     fmt="[%(asctime)s][%(name)s] %(message)s")

    def run(self, model_set, gconfig):
        print(0)
        self.logger.info("Retraining all Models.")
        for model in model_set:
            if(model in gconfig['train_models']): 
                m_dict = model_set[model]
                check = m_dict['model'].train(m_dict)
                # if(check != "OK"): kill_me(check) 
        time.sleep(10)