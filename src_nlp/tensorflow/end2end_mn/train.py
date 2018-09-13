from config import args
from data import DataLoader
from model import model_fn

import tensorflow as tf
import numpy as np
import json
import pprint


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    print(json.dumps(args, indent=4))

    train_dl = DataLoader(
        path='../temp/qa5_three-arg-relations_train.txt',
        is_training=True)
    test_dl = DataLoader(
        path='../temp/qa5_three-arg-relations_test.txt',
        is_training=False, vocab=train_dl.vocab, params=train_dl.params)

    model = tf.estimator.Estimator(model_fn, params=train_dl.params)
    model.train(train_dl.input_fn())
    gen = model.predict(test_dl.input_fn())
    preds = np.concatenate(list(gen))
    preds = np.reshape(preds, [test_dl.data['size'], 2])
    print('Testing Accuracy:', (test_dl.data['val']['answers'][:, 0] == preds[:, 0]).mean())
    demo(test_dl.demo, test_dl.vocab['idx2word'], preds)


def demo(demo, idx2word, ids, demo_idx=3):
    demo_i, demo_q, demo_a = demo
    print()
    pprint.pprint(demo_i[demo_idx])
    print()
    print('Question:', demo_q[demo_idx])
    print()
    print('Ground Truth:', demo_a[demo_idx])
    print()
    print('- '*12)
    print('Machine Answer:', [idx2word[id] for id in ids[demo_idx]])


if __name__ == '__main__':
    main()
