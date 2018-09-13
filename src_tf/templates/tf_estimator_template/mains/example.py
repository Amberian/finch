from sklearn.metrics import classification_report

import tensorflow as tf
import numpy as np
import logging
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from model import model_fn
from data import IMDBDataLoader
from log import create_logging


def main():
    dl = IMDBDataLoader()
    create_logging()
    estimator = tf.estimator.Estimator(model_fn)
    estimator.train(dl.train_input_fn())
    y_pred = np.fromiter(estimator.predict(dl.predict_input_fn()), np.int32)
    tf.logging.info('\n'+classification_report(dl.y_test, y_pred))


if __name__ == '__main__':
    main()