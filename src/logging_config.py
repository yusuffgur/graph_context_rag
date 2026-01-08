import logging
import sys
from logging.handlers import RotatingFileHandler
from src.config import settings

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File (Rotating)
    f_handler = RotatingFileHandler("system.log", maxBytes=5*1024*1024, backupCount=3)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger