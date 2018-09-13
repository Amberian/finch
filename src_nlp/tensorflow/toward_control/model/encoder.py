from base import BaseModel
from configs import args

import tensorflow as tf


class Encoder(BaseModel):
    def __init__(self, vocab):
        super().__init__('Encoder')

        with tf.variable_scope(self._scope):
            self.embedding = tf.get_variable('lookup_table', [vocab.vocab_size, args.embed_dims])
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size,
                                                   kernel_initializer=tf.orthogonal_initializer())
            self.proj_z_mean = tf.layers.Dense(args.latent_size)
            self.proj_z_logvar = tf.layers.Dense(args.latent_size)
    

    def __call__(self, inputs, soft_inp=False):
        with tf.variable_scope(self._scope):
            if soft_inp:
                _inputs = tf.reshape(inputs, [-1, args.vocab_size])
                x = tf.matmul(_inputs, self.embedding)
                batch_sz = tf.shape(inputs)[0]
                x = tf.reshape(x, [batch_sz, args.max_len, args.embed_dims])
            else:
                x = tf.nn.embedding_lookup(self.embedding, inputs)
            _, enc_state = tf.nn.dynamic_rnn(self.rnn_cell,
                                             x,
                                             None if soft_inp else tf.count_nonzero(inputs, 1),
                                             dtype=tf.float32)
            z_mean = self.proj_z_mean(enc_state)
            z_logvar = self.proj_z_logvar(enc_state)
            return z_mean, z_logvar
