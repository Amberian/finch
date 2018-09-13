import tensorflow as tf
import numpy as np

from configs import args


def word_dropout(x, vocab):
    is_dropped = np.random.binomial(1, args.word_dropout_rate, x.shape)
    fn = np.vectorize(lambda x, k: vocab.word2idx['<unk>'] if (
        k and (x not in range(4))) else x)
    return fn(x, is_dropped)


def pipeline_train(enc_inp, dec_inp, dec_out, sess):
    dataset = tf.data.Dataset.from_tensor_slices((enc_inp, dec_inp, dec_out))
    dataset = dataset.shuffle(len(enc_inp)).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()

    enc_inp_ph = tf.placeholder(tf.int32, [None, args.max_len])
    dec_inp_ph = tf.placeholder(tf.int32, [None, args.max_len+1])
    dec_out_ph = tf.placeholder(tf.int32, [None, args.max_len+1])

    init_dict = {enc_inp_ph: enc_inp,
                 dec_inp_ph: dec_inp,
                 dec_out_ph: dec_out}
                 
    sess.run(iterator.initializer, init_dict)
    
    return iterator, init_dict


class WakeSleepDataLoader(object):
    def __init__(self, sess, vocab):
        self.sess = sess
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=args.vocab_size, index_from=4)
        
        X = np.concatenate((X_train, X_test))

        X = tf.keras.preprocessing.sequence.pad_sequences(
                X, args.max_len+1, truncating='pre', padding='post')

        enc_inp = X[:, 1:]
        dec_inp = word_dropout(X, vocab)
        dec_out = np.concatenate([X[:, 1:], np.full([X.shape[0], 1], vocab.word2idx['<end>'])], 1)

        self.train_iterator, self.train_init_dict = pipeline_train(
            enc_inp, dec_inp, dec_out, sess)
