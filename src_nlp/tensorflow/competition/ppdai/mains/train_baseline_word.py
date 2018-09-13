from sklearn.metrics import log_loss

import tensorflow as tf
import numpy as np
import pandas as pd
import pprint

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from log import create_logging
from configs.general import args
from data.dataloaders.dataloader_word_rnn import DataLoader
from model.baseline import model_fn


MODEL_PATH = '../model/baseline_word_ckpt'
SUBMIT_PATH = './submit_baseline_word.csv'


def get_val_labels():
    ori_train_csv = pd.read_csv('../data/files_original/train.csv')
    thres = int(len(ori_train_csv) * args.train_val_split)
    val_csv = ori_train_csv[thres:]
    return val_csv['label'].values


def main():
    create_logging()
    tf.logging.info('\n'+pprint.pformat(args.__dict__))
    dl = DataLoader()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    prev_models = os.listdir(MODEL_PATH)
    if prev_models is not None:
        for f in prev_models:
            os.remove(os.path.join(MODEL_PATH, f))
    
    estimator = tf.estimator.Estimator(model_fn,
                                       model_dir=MODEL_PATH,
                                       config=tf.estimator.RunConfig(keep_checkpoint_max=1))
    
    y_true = get_val_labels()
    for _ in range(args.n_epochs):
        estimator.train(lambda: dl.train_input_fn())
        y_pred = list(estimator.predict(lambda: dl.val_input_fn()))
        tf.logging.info('\nVal Log Loss: %.3f\n' % log_loss(
            np.asarray(y_true, np.float64),
            np.asarray(y_pred, np.float64),
            labels=[0, 1]))
    submit_arr = np.asarray(list(estimator.predict(lambda: dl.predict_input_fn())))
    print(submit_arr.shape)
    
    submit = pd.DataFrame()
    submit['y_pre'] = submit_arr
    submit.to_csv(SUBMIT_PATH, index=False)


if __name__ == '__main__':
    main()