import tensorflow as tf

from configs import args


class IMDBDataLoader(object):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.imdb.load_data(num_words=args.vocab_size)
        self.X_train = tf.keras.preprocessing.sequence.pad_sequences(self.X_train, args.max_len)
        self.X_test = tf.keras.preprocessing.sequence.pad_sequences(self.X_test, args.max_len)


    def train_input_fn(self):
        return tf.estimator.inputs.numpy_input_fn(
            x = self.X_train,
            y = self.y_train,
            batch_size = args.batch_size,
            shuffle = True)


    def predict_input_fn(self):
        return tf.estimator.inputs.numpy_input_fn(
            x = self.X_test,
            shuffle = False)
