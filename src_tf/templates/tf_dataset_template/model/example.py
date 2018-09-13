from configs import args
from base import BaseModel

import tensorflow as tf


def forward(x, reuse, is_training):
    with tf.variable_scope('model', reuse=reuse):
        x = tf.contrib.layers.embed_sequence(x, args.vocab_size, args.embed_dim)
        x = tf.layers.dropout(x, 0.2, training=is_training)
        feat_map = []
        for k_size in [3, 4, 5]:
            _x = tf.layers.conv1d(x, args.filters, k_size, activation=tf.nn.relu)
            _x = tf.layers.max_pooling1d(_x, _x.get_shape().as_list()[1], 1)
            _x = tf.reshape(_x, (tf.shape(x)[0], args.filters))
            feat_map.append(_x)
        x = tf.concat(feat_map, -1)
        x = tf.layers.dense(x, args.filters, tf.nn.relu)
        logits = tf.layers.dense(x, args.n_class)
    return logits


class Model(BaseModel):
    def __init__(self, dataloader):
        super().__init__()
        self.build_train_graph(dataloader)
        self.build_predict_graph(dataloader)


    def build_train_graph(self, dataloader):
        X_batch, y_batch = dataloader.train_iterator.get_next()

        logits = forward(X_batch, reuse=False, is_training=True)

        self.ops['global_step'] = tf.Variable(0, trainable=False)

        self.ops['lr'] = tf.train.exponential_decay(5e-3, self.ops['global_step'], 1400, 0.2)

        self.ops['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y_batch))

        self.ops['train'] = tf.train.AdamOptimizer(self.ops['lr']).minimize(
            self.ops['loss'], global_step=self.ops['global_step'])


    def build_predict_graph(self, dataloader):
        self.ops['pred_logits'] = forward(dataloader.predict_iterator.get_next(),
                                          reuse=True,
                                          is_training=False)
