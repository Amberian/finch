from base import BaseTrainer
from configs import args

import tensorflow as tf
import numpy as np


class Trainer(BaseTrainer):
    def __init__(self, sess, model, dataloader):
        super().__init__(sess, model, dataloader)


    def train(self):
        for epoch in range(1, args.n_epochs+1):
            while True:
                try:
                    _, step, loss, lr = self.sess.run([self.model.ops['train'],
                                                       self.model.ops['global_step'],
                                                       self.model.ops['loss'],
                                                       self.model.ops['lr']])
                except tf.errors.OutOfRangeError:
                    break
                else:
                    if step % args.display_step == 0 or step == 1:
                        tf.logging.info("Epoch %d | Step %d | Loss %.3f | LR: %.4f" % (epoch, step, loss, lr))

            y_pred_li = []
            while True:
                try:
                    y_pred_li.append(self.sess.run(self.model.ops['pred_logits']))
                except tf.errors.OutOfRangeError:
                    break
            y_pred = np.argmax(np.vstack(y_pred_li), 1)
            tf.logging.info("Epoch %d | Validation Accuracy: %.4f" % (
                            epoch, (y_pred==self.dataloader.y_test).mean()))
            
            if epoch != args.n_epochs:
                self.sess.run(self.dataloader.train_iterator.initializer,
                              self.dataloader.train_init_dict)
                self.sess.run(self.dataloader.predict_iterator.initializer,
                              self.dataloader.predict_init_dict)
