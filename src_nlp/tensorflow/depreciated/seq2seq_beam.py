from base_seq2seq import BaseSeq2Seq

import tensorflow as tf
import numpy as np


class Seq2Seq(BaseSeq2Seq):
    def __init__(self, rnn_size, n_layers, X_word2idx, encoder_embedding_dim, Y_word2idx, decoder_embedding_dim,
                 sess=tf.Session(), grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.Y_word2idx = Y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim
        self.beam_width = beam_width
        self.force_teaching_ratio = force_teaching_ratio
        self.sess = sess
        self.register_symbols()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.X)[0]
    # end method add_input_layer


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method lstm_cell


    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.X_word2idx), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))            
        _, self.encoder_state = tf.nn.dynamic_rnn(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
            inputs = tf.nn.embedding_lookup(encoder_embedding, self.X),
            sequence_length = self.X_seq_len,
            dtype = tf.float32)
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.n_layers))
    # end method add_encoder_layer
    

    def processed_decoder_input(self):
        main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input
    # end method add_decoder_layer


    def add_decoder_layer(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [len(self.Y_word2idx), self.decoder_embedding_dim],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length = self.Y_seq_len,
                embedding = decoder_embedding,
                sampling_probability = 1 - self.force_teaching_ratio,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = tf.layers.Dense(len(self.Y_word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output

        with tf.variable_scope('decode', reuse=True):
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
                end_token = self._y_eos,
                initial_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width),
                beam_width = self.beam_width,
                output_layer = tf.layers.Dense(len(self.Y_word2idx), _reuse=True),
                length_penalty_weight = 0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = False,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]
    # end method add_decoder_layer


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path
# end class