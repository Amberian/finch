from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np


class Image2Seq:
    def __init__(self, img_size, word2idx,
                 img_ch=3, data_format='channels_last', kernel_size=(5,5), pool_size=(2,2), padding='valid',
                 embedding_dim=256, rnn_size=256, n_layers=2, grad_clip=5.,
                 sess=tf.Session()):
        self.img_size = img_size
        self.word2idx = word2idx
        self.embedding_dim = embedding_dim

        self.img_ch = img_ch
        self.data_format = data_format
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding

        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip

        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder()
        self.add_decoder()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.img_ch, self.img_size[0], self.img_size[1]])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.train_flag = tf.placeholder(tf.bool)
        self.batch_size = tf.shape(self.X)[0]
        self._pointer = tf.transpose(self.X, [0, 2, 3, 1])
    # end method


    def add_encoder(self):
        self.add_input_layer()
        self.add_conv(32)
        self.add_conv(32)
        self.add_pooling()
        self.add_conv(32)
        self.add_conv(32)
        self.add_pooling()
        self.add_conv(32)
        self.add_conv(32)
        self.add_pooling()
        self.add_projection(self.rnn_size)
    # end method


    def add_decoder(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [len(self.word2idx), self.embedding_dim],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length = self.Y_seq_len-1,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = core_layers.Dense(len(self.word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len-1))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant([self.word2idx['<start>']], dtype=tf.int32), [self.batch_size]),
                end_token = self.word2idx['<end>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
                helper = predicting_helper,
                initial_state = self.encoder_state,
                output_layer = core_layers.Dense(len(self.word2idx), _reuse=True))
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.Y_seq_len-1))
            self.predicting_ids = predicting_decoder_output.sample_id
    # end method


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len-1, tf.reduce_max(self.Y_seq_len-1), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits = self.training_logits, targets = self.processed_decoder_output(), weights = masks)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method

    def processed_decoder_input(self):
        return tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
    # end method

    def processed_decoder_output(self):
        return tf.strided_slice(self.Y, [0, 1], [self.batch_size, tf.shape(self.Y)[1]], [1, 1]) # remove first char
    # end method

    def add_conv(self, out_dim, strides=(1, 1)):
        Y = tf.layers.conv2d(inputs = self._pointer,
                             filters = out_dim,
                             kernel_size = self.kernel_size,
                             strides = strides,
                             padding = self.padding,
                             use_bias = True,
                             activation = tf.nn.relu,
                             data_format = self.data_format)
        self._pointer = tf.layers.batch_normalization(Y, training=self.train_flag)
    # end method


    def add_pooling(self):
        self._pointer = tf.layers.max_pooling2d(inputs = self._pointer,
                                                pool_size = self.pool_size,
                                                strides = self.pool_size,
                                                padding = self.padding,
                                                data_format = self.data_format)
    # end method


    def add_projection(self, out_dim):
        flat = tf.contrib.layers.flatten(self._pointer)
        proj = tf.layers.dense(flat, out_dim)
        self.encoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(c=proj, h=proj) for _ in range(self.n_layers)])
    # end method


    def partial_fit(self, images, captions, lengths):
        _, loss = self.sess.run([self.train_op, self.loss],
            {self.X: images, self.Y: captions, self.Y_seq_len: lengths, self.train_flag:True})
        return loss
    # end method


    def infer(self, image, idx2word):
        out_indices = self.sess.run(self.predicting_ids,
            {self.X:image, self.Y_seq_len:[20], self.train_flag:False})[0]
        print('{}'.format(' '.join([idx2word[i] for i in out_indices])))
    # end method
# end class
