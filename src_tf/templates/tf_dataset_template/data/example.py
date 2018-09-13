import tensorflow as tf

from configs import args


def pipeline_train(X, y, sess):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, args.max_len])
    y_ph = tf.placeholder(tf.int64, [None])
    init_dict = {X_ph: X, y_ph: y}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict


def pipeline_test(X, sess):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, args.max_len])
    init_dict = {X_ph: X}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict


class IMDBDataLoader(object):
    def __init__(self, sess):
        self.sess = sess

        (X_train, y_train), (X_test, self.y_test) = tf.keras.datasets.imdb.load_data(num_words=args.vocab_size)
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, args.max_len)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, args.max_len)

        self.train_iterator, self.train_init_dict = pipeline_train(X_train, y_train, sess)
        self.predict_iterator, self.predict_init_dict = pipeline_test(X_test, sess)
