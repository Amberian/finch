from base import BaseModel
from configs import args
from utils import ModifiedBasicDecoder, ModifiedBeamSearchDecoder

import tensorflow as tf


class Generator(BaseModel):
    def __init__(self, vocab):
        super().__init__('Generator')
        self.vocab = vocab

        with tf.variable_scope(self._scope):
            self.embedding = tf.get_variable('lookup_table', [vocab.vocab_size, args.embed_dims])
            self.dec_cell = tf.nn.rnn_cell.GRUCell(args.rnn_size,
                                                   kernel_initializer=tf.orthogonal_initializer())
            self.state_proj = tf.layers.Dense(args.rnn_size, tf.nn.elu)
            self.output_proj = tf.layers.Dense(vocab.vocab_size, _scope='decoder/output_proj')
    

    def __call__(self, latent_vec, is_training, dec_inp=None):
        with tf.variable_scope(self._scope):
            init_state = self.state_proj(latent_vec)
            batch_sz = tf.shape(init_state)[0]

            if is_training:
                dec_seq_len = tf.count_nonzero(dec_inp, 1, dtype=tf.int32)
            
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs = tf.nn.embedding_lookup(self.embedding, dec_inp),
                    sequence_length = dec_seq_len)
                decoder = ModifiedBasicDecoder(
                    cell = self.dec_cell,
                    helper = helper,
                    initial_state = init_state,
                    concat_z = latent_vec)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    maximum_iterations = args.max_len + 1)
                rnn_output = decoder_output.rnn_output
                return rnn_output, self.output_proj(rnn_output)
            else:
                decoder = ModifiedBeamSearchDecoder(
                    cell = self.dec_cell,
                    embedding = self.embedding,
                    start_tokens = tf.tile(tf.constant([self.vocab.word2idx['<start>']], tf.int32),
                                           [batch_sz]),
                    end_token = self.vocab.word2idx['<end>'],
                    initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                    beam_width = args.beam_width,
                    output_layer = self.output_proj,
                    concat_z = tf.tile(tf.expand_dims(latent_vec, 1), [1, args.beam_width, 1]))
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder,
                    maximum_iterations = args.max_len + 1)
                return decoder_output.predicted_ids[:, :, 0]
