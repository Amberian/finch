import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, dataloader):
        self.sess = sess        
        self.model = model
        self.dataloader = dataloader
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
    

    def train(self):
        raise NotImplementedError
