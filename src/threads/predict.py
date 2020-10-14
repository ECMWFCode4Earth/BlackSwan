import threading
import logging, logging.config

class PredictThread(threading.Thread):
    def __init__(self, threadID):
        self.threadID = threadID
        self.threadName = "PredictThread"
        logging.config.fileConfig(fname='log_config.ini', disable_existing_loggers=False)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG', logger=self.logger,fmt="[%(asctime)s][%(name)s] %(message)s",)

    def run(self, model_set, gconfig):
        logger.info("Evaluating all Models.")
        for model in model_set:
            m_dict = model_set[model]
            if(model in gconfig['PRED_MODELS']):
                m_dict['model'].predict(m_dict)
