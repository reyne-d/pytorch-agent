import logging
import logging.config
import sys
import os


def setup_logger(logger_name=None, log_path='log.txt',
                 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 filemode='w', level=logging.DEBUG):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name) # if logger_name is None, return root logger
    formatter = logging.Formatter(format)

    # add filehandler, so that log can be saved in txt
    filehandler = logging.FileHandler(filename=log_path, mode=filemode)
    filehandler.setFormatter(formatter)

    # add streamHandler, so that log can print in screen
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    # the Handler.setLeval can not work
    logger.setLevel(level)

    logger.info("Log will be saved in  (%s)." % (log_path))
    return logger