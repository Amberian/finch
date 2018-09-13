import tensorflow as tf
import pprint
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from model import VAE
from data.imdb import VAEDataLoader
from vocab.imdb import IMDBVocab
from trainers import VAETrainer
from log import create_logging


def main():
    create_logging()
    sess = tf.Session()
    vocab = IMDBVocab()
    dl = VAEDataLoader(sess, vocab)

    model = VAE(dl, vocab)
    tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
    trainer = VAETrainer(sess, model, dl, vocab)
    trainer.train()


if __name__ == '__main__':
    main()