from datetime import datetime

import logging
import os


def create_logging():
    files = os.listdir('../log')
    for f in files:
        if f.endswith('.log'):
            os.remove('../log/'+f)

    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler('../log/'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
