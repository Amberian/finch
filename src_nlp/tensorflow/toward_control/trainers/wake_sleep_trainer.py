from base import BaseTrainer
from configs import args

import tensorflow as tf
import numpy as np


def inf_inp(test_strs, vocab):
    x = [[vocab.word2idx.get(w, vocab.word2idx['<unk>']) for w in s.split()] for s in test_strs]
    x = tf.keras.preprocessing.sequence.pad_sequences(
        x, args.max_len, truncating='post', padding='post')
    return x


def demo(test_strs, pred_ids, reve_ids, vocab):
    for s, pred, reve in zip(test_strs, pred_ids, reve_ids):
        tf.logging.info('\nOriginal: '+s+'\n')
        tf.logging.info('\nReconstr: '+(' '.join([vocab.idx2word.get(idx, '<unk>') for idx in pred]))+'\n')
        tf.logging.info('\nReversed: '+(' '.join([vocab.idx2word.get(idx, '<unk>') for idx in reve]))+'\n')


class WakeSleepTrainer(BaseTrainer):
    def __init__(self, sess, model, discri_dl, wake_sleep_dl, vocab):
        super().__init__(sess, model)
        self.discri_dl = discri_dl
        self.wake_sleep_dl = wake_sleep_dl
        self.vocab = vocab


    def train(self):
        for epoch in range(1, args.n_epochs+1):
            while True:
                try:
                    (_, step, d_loss, clf_loss, L_u, entropy, clf_acc) = self.sess.run(
                        [self.model.ops['discri']['train'],
                         self.model.ops['global_step'],
                         self.model.ops['discri']['loss'],
                         self.model.ops['discri']['clf_loss'],
                         self.model.ops['discri']['L_u'],
                         self.model.ops['discri']['entropy'],
                         self.model.ops['discri']['clf_acc'],
                    ])

                    (_, e_loss, kl_w, kl_loss, nll_loss,
                     _, g_loss, temper, l_attr_c, l_attr_z) = self.sess.run(
                        [self.model.ops['encoder']['train'],
                         self.model.ops['encoder']['loss'],
                         self.model.ops['vae']['kl_w'],
                         self.model.ops['vae']['kl_loss'],
                         self.model.ops['vae']['nll_loss'],
                         self.model.ops['generator']['train'],
                         self.model.ops['generator']['loss'],
                         self.model.ops['generator']['temperature'],
                         self.model.ops['generator']['l_attr_c'],
                         self.model.ops['generator']['l_attr_z']
                    ])
                except tf.errors.OutOfRangeError:
                    break
                else:
                    if step % args.wake_sleep_display_step == 0 or step == 1:
                        tf.logging.info("\nDiscriminator | Epoch [%d/%d] | Step %d | loss: %.3f\nclf_loss: %.3f | clf_acc: %.3f | L_u: %.3f | entropy: %.3f\n" % (
                            epoch, args.n_epochs, step, d_loss, clf_loss, clf_acc, L_u, entropy))
                
                        tf.logging.info("\nEncoder | Epoch [%d/%d] | Step %d | loss: %.3f\nkl_w: %.3f | kl_loss: %.3f | nll_loss: %.3f\n" % (
                            epoch, args.n_epochs, step, e_loss, kl_w, kl_loss, nll_loss))

                        tf.logging.info("\nGenerator | Epoch [%d/%d] | Step %d | loss: %.3f\ntemper:%.3f | l_attr_c: %.3f | l_attr_z: %.3f \n" % (
                            epoch, args.n_epochs, step, g_loss, temper, l_attr_c, l_attr_z))
                    
                    if step % 10 == 0 or step == 1:
                        test_strs = ['i love this film and i think it is one of the best films',
                                     'this movie is a waste of time and there is no point to watch it']
                        pred_ids, reve_ids = self.sess.run(
                            [self.model.ops['infe_ids']['direct'],
                             self.model.ops['infe_ids']['reversed']],
                            {self.model.ops['infe_ph']: inf_inp(test_strs, self.vocab)})
                        demo(test_strs, pred_ids, reve_ids, self.vocab)
                
            if epoch != args.n_epochs:
                self.sess.run(self.discri_dl.train_iterator.initializer,
                              self.discri_dl.train_init_dict)
                self.sess.run(self.wake_sleep_dl.train_iterator.initializer,
                              self.wake_sleep_dl.train_init_dict)
