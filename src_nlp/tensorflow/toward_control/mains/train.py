import tensorflow as tf
import pprint
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from configs import args
from model import WakeSleepController
from data.imdb import WakeSleepDataLoader, DiscriminatorDataLoader
from vocab.imdb import IMDBVocab
from trainers import WakeSleepTrainer
from log import create_logging


def main():
    create_logging()
    tf.logging.info('\n'+pprint.pformat(args.__dict__))
    sess = tf.Session()
    vocab = IMDBVocab()
    discri_dl = DiscriminatorDataLoader(sess, vocab)
    wake_sleep_dl = WakeSleepDataLoader(sess, vocab)

    model = WakeSleepController(discri_dl, wake_sleep_dl, vocab)
    tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
    trainer = WakeSleepTrainer(sess, model, discri_dl, wake_sleep_dl, vocab)
    model.load(sess, args.vae_ckpt_dir)
    trainer.train()


if __name__ == '__main__':
    main()