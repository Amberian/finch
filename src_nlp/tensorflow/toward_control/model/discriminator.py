from base import BaseModel
from configs import args

import tensorflow as tf


class _Discriminator(BaseModel):
    def __init__(self, build_graph=True):
        super().__init__('Discriminator')

        with tf.variable_scope(self._scope):
            self.embedding = tf.get_variable('lookup_table', [args.vocab_size, args.embed_dims])
            self.dropout_embed = tf.layers.Dropout(args.discriminator_dropout_rate)
            self.attn_proj = tf.layers.Dense(1, tf.tanh)
            self.output_proj = tf.layers.Dense(args.n_class)
        
        if build_graph:
            self.build_train_graph(dataloader)
            self.build_predict_graph(dataloader)
    

    def __call__(self, inputs, is_training, soft_inp=False):
        with tf.variable_scope(self._scope):
            if soft_inp:
                _inputs = tf.reshape(inputs, [-1, args.vocab_size])
                x = tf.matmul(_inputs, self.embedding)
                batch_sz = tf.shape(inputs)[0]
                x = tf.reshape(x, [batch_sz, args.max_len, args.embed_dims])
            else:
                x = tf.nn.embedding_lookup(self.embedding, inputs)
            
            x = self.dropout_embed(x, training=is_training)
            align = tf.squeeze(self.attn_proj(x), -1)
            align = tf.expand_dims(tf.nn.softmax(align), -1)
            x = tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)
            logits = self.output_proj(x)
            return logits
    


class Discriminator(BaseModel):
    def __init__(self, build_graph=True):
        super().__init__('Discriminator')

        with tf.variable_scope(self._scope):
            self.embedding = tf.get_variable('lookup_table', [args.vocab_size, args.embed_dims])
            self.dropout_embed = tf.layers.Dropout(args.discriminator_dropout_rate)
            self.conv_k3 = tf.layers.Conv1D(args.n_filters, 3, activation=tf.nn.relu)
            self.conv_k4 = tf.layers.Conv1D(args.n_filters, 4, activation=tf.nn.relu)
            self.conv_k5 = tf.layers.Conv1D(args.n_filters, 5, activation=tf.nn.relu)
            self.dropout_feat = tf.layers.Dropout(args.discriminator_dropout_rate)
            self.hidden_proj = tf.layers.Dense(args.n_filters, tf.nn.relu)
            self.output_proj = tf.layers.Dense(args.n_class)
        
        if build_graph:
            self.build_train_graph(dataloader)
            self.build_predict_graph(dataloader)
    

    def __call__(self, inputs, is_training, soft_inp=False):
        with tf.variable_scope(self._scope):
            if soft_inp:
                _inputs = tf.reshape(inputs, [-1, args.vocab_size])
                x = tf.matmul(_inputs, self.embedding)
                batch_sz = tf.shape(inputs)[0]
                x = tf.reshape(x, [batch_sz, args.max_len, args.embed_dims])
            else:
                x = tf.nn.embedding_lookup(self.embedding, inputs)
            
            x = self.dropout_embed(x, training=is_training)
            feat_map = []
            for conv in [self.conv_k3, self.conv_k4, self.conv_k5]:
                _x = conv(x)
                _x = tf.layers.max_pooling1d(_x, _x.get_shape().as_list()[1], 1)
                _x = tf.reshape(_x, (tf.shape(x)[0], args.n_filters))
                feat_map.append(_x)
            
            x = tf.concat(feat_map, -1)
            x = self.dropout_feat(x, training=is_training)
            x = self.hidden_proj(x)
            logits = self.output_proj(x)
            return logits

    
    def build_train_graph(self, dataloader):
        X_batch, y_batch = dataloader.train_iterator.get_next()

        logits = self.forward(X_batch, is_training=True)

        self.ops['global_step'] = tf.Variable(0, trainable=False)

        self.ops['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y_batch))

        self.ops['train'] = tf.train.AdamOptimizer().minimize(
            self.ops['loss'], global_step=self.ops['global_step'])


    def build_predict_graph(self, dataloader):
        self.ops['pred_logits'] = self.forward(dataloader.predict_iterator.get_next(),
                                               is_training=False)
    