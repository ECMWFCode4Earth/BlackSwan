import threading
import logging, logging.config

class TrainThread(threading.Thread):
    def __init__(self, threadID):
        self.threadID = threadID
        self.threadName = "TrainThread"
        logging.config.fileConfig(fname='log_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG', logger=self.logger,fmt="[%(asctime)s][%(name)s] %(message)s",)

    def run(self, model_set, gconfig):
        self.logger.info("Retraining all Models.")
        for model in model_set:
            m_dict = model_set[model]
            if(model in gconfig['TRAIN_MODELS']):  
                m_dict['model'].train(m_dict)
