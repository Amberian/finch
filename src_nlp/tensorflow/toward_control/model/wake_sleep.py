from base import BaseModel
from configs import args

from .vae import VAE
from .discriminator import Discriminator

import tensorflow as tf
import os


class WakeSleepController(BaseModel):
    def __init__(self, discri_dl, wake_sleep_dl, vocab):
        super().__init__('WakeSleepController')
        self.vae = VAE(dataloader=None, vocab=vocab, build_graph=False)
        self.discriminator = Discriminator(build_graph=False)
        self.ops = {'discri': {},
                    'vae': {},
                    'encoder': {},
                    'generator': {},
                    'infe_ids': {'direct': None,
                                 'reversed': None}}

        self.ops['global_step'] = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        
        enc_inp, labels = discri_dl.train_iterator.get_next()
        self.build_discriminator_graph(enc_inp, labels)

        enc_inp, dec_inp, dec_out = wake_sleep_dl.train_iterator.get_next()
        self.build_encoder_generator_graph(enc_inp, dec_inp, dec_out)
        self.build_inference_graph()
        self.init_saver()

    
    def build_discriminator_graph(self, enc_inp, labels):
        batch_sz = tf.shape(enc_inp)[0]
        
        logits_real = self.discriminator(enc_inp, is_training=True)
        self.ops['discri']['clf_loss'] = self.sparse_cross_entropy_fn(logits_real, labels)
        acc = tf.equal(tf.argmax(logits_real, -1), labels)
        self.ops['discri']['clf_acc'] = tf.reduce_mean(tf.to_float(acc))

        c_prior = self.vae.draw_c_prior(batch_sz)
        z_prior = self.vae.draw_z_prior(batch_sz)                
        latent_vec = tf.concat((z_prior, c_prior), -1)
        ids_gen = self.vae.generator(latent_vec, is_training=False)
        ids_gen = tf.reshape(ids_gen, [batch_sz, args.max_len+1])

        logits_fake = self.discriminator(ids_gen, is_training=True)
        self.ops['discri']['entropy'] = - tf.reduce_sum(tf.nn.log_softmax(logits_fake))
        self.ops['discri']['L_u'] = self.cross_entropy_fn(
            logits_fake, c_prior) + args.beta * self.ops['discri']['entropy']

        self.ops['discri']['loss'] = self.ops['discri']['clf_loss'] + args.lambda_u * self.ops['discri']['L_u']
        self.ops['discri']['train'] = self.optimizer.minimize(
            self.ops['discri']['loss'],
            var_list = tf.trainable_variables(self.discriminator._scope),
            global_step = self.ops['global_step'])


    def build_encoder_generator_graph(self, enc_inp, dec_inp, dec_out):
        batch_sz = tf.shape(enc_inp)[0]

        z_mean, z_logvar = self.vae.encoder(enc_inp)
        z = self.vae.reparam_trick(z_mean, z_logvar)
        c = self.vae.draw_c(self.discriminator(enc_inp, is_training=False))
        latent_vec = tf.concat((z, c), -1)
        rnn_output, _ = self.vae.generator(latent_vec, is_training=True, dec_inp=dec_inp)
        self.build_vae_loss(rnn_output, z_mean, z_logvar, dec_out)

        z_prior = self.vae.draw_z_prior(batch_sz)
        c_prior = self.vae.draw_c_prior(batch_sz)
        latent_vec = tf.concat((z_prior, c_prior), -1)
        _, logits_gen = self.vae.generator(latent_vec, is_training=True, dec_inp=dec_inp)

        temper = self.temperature_fn()
        self.ops['generator']['temperature'] = tf.cond(temper<1e-3, lambda: 1e-3, lambda: temper)
        
        x_hat = tf.nn.softmax(logits_gen[:, :-1, :] / self.ops['generator']['temperature'])

        c_logits = self.discriminator(x_hat, is_training=False, soft_inp=True)
        self.ops['generator']['l_attr_c'] = self.cross_entropy_fn(c_logits, c_prior)
        z_mean_gen, z_logvar_gen = self.vae.encoder(x_hat, soft_inp=True)
        self.ops['generator']['l_attr_z'] = self.mutinfo_loss_fn(
            z_mean_gen, z_logvar_gen, z_prior)
        
        self.ops['encoder']['loss'] = self.ops['vae']['loss']
        self.ops['generator']['loss'] = self.ops['vae']['loss'] + (
            args.lambda_c * self.ops['generator']['l_attr_c']) + (
                args.lambda_z * self.ops['generator']['l_attr_z'])

        self.ops['encoder']['train'] = self.optimizer.apply_gradients(
            self.clip_grads(self.ops['encoder']['loss'], self.vae.encoder._scope))
        self.ops['generator']['train'] = self.optimizer.apply_gradients(
            self.clip_grads(self.ops['generator']['loss'], self.vae.generator._scope))
       
    
    def build_inference_graph(self):
        self.ops['infe_ph'] = tf.placeholder(tf.int32, [None, args.max_len])

        z_mean, z_logvar = self.vae.encoder(self.ops['infe_ph'])
        z = self.vae.reparam_trick(z_mean, z_logvar)
        c = self.vae.draw_c(self.discriminator(self.ops['infe_ph'], is_training=False))

        latent_vec = tf.concat((z, c), -1)
        reversed_vec = tf.concat((z, 1-c), -1)

        self.ops['infe_ids']['direct'] = self.vae.generator(latent_vec, is_training=False)
        self.ops['infe_ids']['reversed'] = self.vae.generator(reversed_vec, is_training=False)


    def sparse_cross_entropy_fn(self, logits, labels):
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


    def cross_entropy_fn(self, logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))


    def build_vae_loss(self, rnn_output, z_mean, z_logvar, dec_out):
        self.ops['vae']['kl_w'] = tf.constant(1.0)

        self.ops['vae']['kl_loss'] = self.vae.kl_loss_fn(z_mean, z_logvar)

        with tf.variable_scope('Generator/decoder/output_proj', reuse=True):
            _weights = tf.transpose(tf.get_variable('kernel'))
            _biases = tf.get_variable('bias')

        mask = tf.reshape(tf.to_float(tf.sign(dec_out)), [-1])

        self.ops['vae']['nll_loss'] = tf.reduce_sum(mask * tf.nn.sampled_softmax_loss(
            weights = _weights,
            biases = _biases,
            labels = tf.reshape(dec_out, [-1, 1]),
            inputs = tf.reshape(rnn_output, [-1, args.rnn_size]),
            num_sampled = args.num_sampled,
            num_classes = args.vocab_size,
        )) / tf.to_float(tf.shape(dec_out)[0])

        self.ops['vae']['loss'] = self.ops['vae']['nll_loss'] + self.ops['vae']['kl_w'] * self.ops['vae']['kl_loss']

    
    def clip_grads(self, loss_op, scope):
        params = tf.trainable_variables(scope=scope)
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return zip(clipped_gradients, params)


    def temperature_fn(self):
        inv_sigmoid = lambda x: 1 / (1 + tf.exp(x))
        return args.temp_anneal_max * inv_sigmoid((10 / args.temp_anneal_bias) * (
            tf.to_float(self.ops['global_step']) - tf.constant(args.temp_anneal_bias / 2)))


    def mutinfo_loss_fn(self, z_mean_new, z_logvar_new, z_prior):
        dist = tf.contrib.distributions.MultivariateNormalDiag(z_mean_new,
                                                               tf.exp(z_logvar_new),
                                                               validate_args=True)
        mutinfo_loss = - dist.log_prob(z_prior)
        return tf.reduce_sum(mutinfo_loss) / tf.to_float(tf.shape(z_prior)[1])


    def init_saver(self):
        enc_vars = tf.trainable_variables(self.vae.encoder._scope)
        gen_vars = tf.trainable_variables(self.vae.generator._scope)
        self.saver = tf.train.Saver(enc_vars+gen_vars, max_to_keep=1)
        p = os.path.dirname(args.vae_ckpt_dir)
        if not os.path.exists(p):
            os.makedirs(p)
    
