import logging
import os
from datetime import datetime

# defining the log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# defining the log path
log_path = os.path.join(os.getcwd(),"logs")

# creating the directory for log
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILE_PATH,
                    format='[%(asctime)s] - %(lineno)d  %(name)s - %(levelname)s- %(message)s')

if __name__=="__main__":
    logging.info("here i am starting the logging")