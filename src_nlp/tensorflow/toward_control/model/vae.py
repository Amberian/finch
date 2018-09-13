from base import BaseModel
from configs import args
from .encoder import Encoder
from .generator import Generator

import tensorflow as tf
import os


class VAE(BaseModel):
    def __init__(self, dataloader, vocab, build_graph=True):
        super().__init__('VAE')
        self.vocab = vocab
        self.encoder = Encoder(vocab)
        self.generator = Generator(vocab)
        if build_graph:
            self.build_train_graph(dataloader)
            self.init_saver()
            self.build_inference_graph()

    
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        p = os.path.dirname(args.vae_ckpt_dir)
        if not os.path.exists(p):
            os.makedirs(p)


    def draw_z_prior(self, batch_sz):
        return tf.truncated_normal([batch_sz, args.latent_size])


    def draw_c_prior(self, batch_sz):
        return tf.contrib.distributions.OneHotCategorical(
            logits=tf.ones([batch_sz, args.n_class]), dtype=tf.float32).sample()


    def draw_c(self, logits):
        return tf.contrib.distributions.OneHotCategorical(logits=logits, dtype=tf.float32).sample()


    def reparam_trick(self, z_mean, z_logvar):
        z = z_mean + tf.exp(0.5 * z_logvar) * self.draw_z_prior(tf.shape(z_mean)[0])
        return z


    def kl_w_fn(self, global_step):
        return args.anneal_max * tf.sigmoid((10 / args.anneal_bias) * (
            tf.to_float(global_step) - tf.constant(args.anneal_bias / 2)))


    def kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(
            tf.exp(gamma) + tf.square(mean) - 1 - gamma) / tf.to_float(tf.shape(mean)[0])

    def clip_grads(self, loss):
        variables = tf.trainable_variables()
        grads = tf.gradients(loss, variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
        return zip(clipped_grads, variables)


    def build_train_graph(self, dataloader):
        enc_inp, dec_inp, dec_out = dataloader.train_iterator.get_next()
        batch_sz = tf.shape(enc_inp)[0]

        z_mean, z_logvar = self.encoder(enc_inp)
        z = self.reparam_trick(z_mean, z_logvar)
        c = self.draw_c_prior(batch_sz)

        latent_vec = tf.concat((z, c), -1)
        rnn_output, _ = self.generator(latent_vec,
                                       is_training = True,
                                       dec_inp = dec_inp)

        self.ops['global_step'] = tf.Variable(0, trainable=False)

        self.ops['kl_w'] = self.kl_w_fn(self.ops['global_step'])

        self.ops['kl_loss'] = self.kl_loss_fn(z_mean, z_logvar)

        with tf.variable_scope('Generator/decoder/output_proj', reuse=True):
            _weights = tf.transpose(tf.get_variable('kernel'))
            _biases = tf.get_variable('bias')

        mask = tf.reshape(tf.to_float(tf.sign(dec_out)), [-1])

        self.ops['nll_loss'] = tf.reduce_sum(mask * tf.nn.sampled_softmax_loss(
            weights = _weights,
            biases = _biases,
            labels = tf.reshape(dec_out, [-1, 1]),
            inputs = tf.reshape(rnn_output, [-1, args.rnn_size]),
            num_sampled = args.num_sampled,
            num_classes = dataloader.vocab_size,
        )) / tf.to_float(batch_sz)

        self.ops['loss'] = self.ops['nll_loss'] + self.ops['kl_w'] * self.ops['kl_loss']
        
        self.ops['train'] = tf.train.AdamOptimizer().apply_gradients(
            self.clip_grads(self.ops['loss']), global_step=self.ops['global_step'])

    def build_inference_graph(self):
        self.ops['infe_ph'] = tf.placeholder(tf.int32, [None, args.max_len])

        z_mean, z_logvar = self.encoder(self.ops['infe_ph'])
        z = self.reparam_trick(z_mean, z_logvar)
        c = self.draw_c_prior(tf.shape(self.ops['infe_ph'])[0])

        latent_vec = tf.concat((z, c), -1)
        self.ops['infe_pred_ids'] = self.generator(latent_vec, is_training=False)
