import tensorflow as tf
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from log import create_logging
from model import Model
from data import IMDBDataLoader
from trainers import Trainer


def main():
    create_logging()
    sess = tf.Session()
    dl = IMDBDataLoader(sess)
    model = Model(dl)
    trainer = Trainer(sess, model, dl)
    trainer.train()


if __name__ == '__main__':
    main()