from base import BaseTrainer
from configs import args

import tensorflow as tf
import numpy as np


def inf_inp(test_strs, vocab):
    x = [[vocab.word2idx.get(w, vocab.word2idx['<unk>']) for w in s.split()] for s in test_strs]
    x = tf.keras.preprocessing.sequence.pad_sequences(
        x, args.max_len, truncating='post', padding='post')
    return x


def demo(test_strs, pred_ids, vocab):
    for s, pred in zip(test_strs, pred_ids):
        tf.logging.info('\nOriginal: '+s+'\n')
        tf.logging.info('\nReconstr: '+(' '.join([vocab.idx2word.get(idx, '<unk>') for idx in pred]))+'\n')


class VAETrainer(BaseTrainer):
    def __init__(self, sess, model, dataloader, vocab):
        super().__init__(sess, model)
        self.dataloader = dataloader
        self.vocab = vocab


    def train(self):
        for epoch in range(1, args.n_epochs+1):
            while True:
                try:
                    _, step, loss, nll_loss, kl_w, kl_loss = self.sess.run(
                        [self.model.ops['train'],
                         self.model.ops['global_step'],
                         self.model.ops['loss'],
                         self.model.ops['nll_loss'],
                         self.model.ops['kl_w'],
                         self.model.ops['kl_loss'],
                    ])
                except tf.errors.OutOfRangeError:
                    break
                else:
                    if step % args.vae_display_step == 0 or step == 1:
                        tf.logging.info("Epoch [%d/%d] | Step %d | Loss %.3f \nnll: %.3f | kl_w: %.3f | kl_loss: %.3f\n" % (
                            epoch, args.n_epochs, step, loss, nll_loss, kl_w, kl_loss))
            
            test_strs = ['i love this film and i think it is one of the best films',
                         'this movie is a waste of time and there is no point to watch it']
            pred_ids = self.sess.run(self.model.ops['infe_pred_ids'],
                                     {self.model.ops['infe_ph']: inf_inp(test_strs, self.vocab)})
            demo(test_strs, pred_ids, self.vocab)

            if epoch != args.n_epochs:
                self.sess.run(self.dataloader.train_iterator.initializer,
                              self.dataloader.train_init_dict)

            self.model.save(self.sess, args.vae_ckpt_dir)
