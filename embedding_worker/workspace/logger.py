import os
from pathlib import Path
import logging
import logging.handlers
import sys

def create_logger_object(filename):
    try:
        logger = logging.getLogger(filename)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [ %(filename)s:%(lineno)s | %(funcName)20s() ] %(message)s')
        if not os.path.exists("./logs"):
            os.mkdir("./logs")

        log_file_name = os.path.join("./logs", filename)
        fh = logging.handlers.RotatingFileHandler(log_file_name, maxBytes = 20971520, backupCount = 8)
        fh.setFormatter(formatter)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.addHandler(fh)
        return logger
    except:
        print("error in creating logger object %s", str(sys.exc_info()))


logger = create_logger_object("embedding_worker.logs")