import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


class Estimator:
    def __init__(self, rnn_size, n_layers, embedding_dims, X_word2idx, Y_word2idx, grad_clip=5.0):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.embedding_dims = embedding_dims
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.Y_word2idx = Y_word2idx
        self.register_symbols()
        self.model = tf.estimator.Estimator(self.model_fn)
    # end constructor


    def seq2seq(self, x_dict, reuse):
        x = x_dict['inputs']
        x_seq_len = x_dict['in_lengths']
        
        with tf.variable_scope('encoder', reuse=reuse):
            encoder_embedding = tf.get_variable('encoder_embedding') if reuse else tf.get_variable(
                'encoder_embedding', [len(self.X_word2idx), self.embedding_dims], tf.float32,
                tf.random_uniform_initializer(-1.0, 1.0))
                        
            _, encoder_state = tf.nn.dynamic_rnn(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]), 
                inputs = tf.nn.embedding_lookup(encoder_embedding, x),
                sequence_length = x_seq_len,
                dtype = tf.float32)
            
            encoder_state = tuple(encoder_state[-1] for _ in range(self.n_layers))

        if not reuse:
            y = x_dict['outputs']
            y_seq_len = x_dict['out_lengths']

            with tf.variable_scope('decoder', reuse=reuse):
                decoder_embedding = tf.get_variable(
                    'decoder_embedding', [len(self.Y_word2idx), self.embedding_dims], tf.float32,
                    tf.random_uniform_initializer(-1.0, 1.0))
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input(y)),
                    sequence_length = y_seq_len,
                    time_major = False)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
                    helper = helper,
                    initial_state = encoder_state,
                    output_layer = tf.layers.Dense(len(self.Y_word2idx)))
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    impute_finished = True,
                    maximum_iterations = tf.reduce_max(y_seq_len))
                return decoder_output.rnn_output
        
        if reuse:
            with tf.variable_scope('decoder', reuse=reuse):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding = tf.get_variable('decoder_embedding'),
                    start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [tf.shape(x)[0]]),
                    end_token = self._y_eos)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = tf.nn.rnn_cell.MultiRNNCell(
                        [self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
                    helper = helper,
                    initial_state = encoder_state,
                    output_layer = tf.layers.Dense(len(self.Y_word2idx), _reuse=reuse))
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    impute_finished = True,
                    maximum_iterations = 2 * tf.reduce_max(x_seq_len))
                return decoder_output.sample_id
    # end method


    def model_fn(self, features, labels, mode):        
        logits = self.seq2seq(features, reuse=False)
        predictions = self.seq2seq(features, reuse=True)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        y_seq_len = features['out_lengths']
        masks = tf.sequence_mask(y_seq_len, tf.reduce_max(y_seq_len), dtype=tf.float32)
        loss_op = tf.contrib.seq2seq.sequence_loss(logits = logits, targets = labels, weights = masks)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        
        train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params),
                                                            global_step=tf.train.get_global_step())
        acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        estim_specs = tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss_op,
            train_op = train_op,
            eval_metric_ops = {'accuracy': acc_op})
        return estim_specs
    # end method


    def fit(self, x, x_seq_len, y, y_seq_len, batch_size=128, n_epoch=10):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'inputs':x, 'in_lengths':x_seq_len, 'outputs':y, 'out_lengths':y_seq_len}, y=y,
            batch_size=batch_size, num_epochs=n_epoch, shuffle=True)
        self.model.train(input_fn)
    # end method


    def infer(self, input_word, X_idx2word, Y_idx2word):
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'inputs': np.atleast_2d(input_indices).astype(np.int32),
               'in_lengths': np.atleast_1d(len(input_indices)).astype(np.int32),
               'outputs': np.atleast_2d(0).astype(np.int32),
               'out_lengths': np.atleast_1d(0).astype(np.int32)},
            shuffle=False)
        out_indices = list(self.model.predict(input_fn))[0]

        print('IN: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        print('OUT: {}'.format(' '.join([Y_idx2word[i] for i in out_indices])))
    # end method


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method


    def processed_decoder_input(self, Y):
        batch_size = tf.shape(Y)[0]
        main = tf.strided_slice(Y, [0, 0], [batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([batch_size, 1], self._y_go), main], 1)
        return decoder_input
    # end method


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = self.Y_word2idx['<PAD>']
        self._y_unk = self.Y_word2idx['<UNK>']
    # end method add_symbols
# end class