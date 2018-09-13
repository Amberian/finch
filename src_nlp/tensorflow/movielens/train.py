from __future__ import print_function
from model import model_fn
from data import DataLoader

import tensorflow as tf
import paddle.v2 as paddle
#tf.logging.set_verbosity(tf.logging.INFO)

NUM_EPOCHS = 50
BATCH_SIZE = 256

def main():
    model = tf.estimator.Estimator(model_fn, params={
        'lr': 1e-4,
        'movie_id_size': paddle.dataset.movielens.max_movie_id() + 1,
        'job_id_size': paddle.dataset.movielens.max_job_id() + 1,
        'user_id_size': paddle.dataset.movielens.max_user_id() + 1,
        'age_id_size': len(paddle.dataset.movielens.age_table),
        'movie_title_vocab_size': 5175,
    })
    dl = DataLoader(BATCH_SIZE)
    for i in xrange(NUM_EPOCHS):
        model.train(dl.train_pipeline())
        print('Testing loss:', model.evaluate(dl.eval_pipeline())['mse'])

if __name__ == '__main__':
    main()
