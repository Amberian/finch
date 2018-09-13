import tensorflow as tf
import numpy as np

from configs import args


MAXLEN = 400


def pipeline_train(enc_inp, labels, sess):
    dataset = tf.data.Dataset.from_tensor_slices((enc_inp, labels))
    dataset = dataset.shuffle(len(labels)).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()

    enc_inp_ph = tf.placeholder(tf.int32, [None, MAXLEN])
    labels_ph = tf.placeholder(tf.int32, [None])

    init_dict = {enc_inp_ph: enc_inp, labels_ph: labels}
                 
    sess.run(iterator.initializer, init_dict)
    
    return iterator, init_dict


class DiscriminatorDataLoader(object):
    def __init__(self, sess, vocab):
        self.sess = sess
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=args.vocab_size, index_from=4)
        
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        X = tf.keras.preprocessing.sequence.pad_sequences(
            X, MAXLEN, truncating='pre', padding='post')

        self.train_iterator, self.train_init_dict = pipeline_train(X, y, sess)
