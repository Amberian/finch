from configs import args

import tensorflow as tf


def forward(x, mode):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

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


def model_fn(features, labels, mode):
    logits = forward(features, mode)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        preds = tf.argmax(logits, -1)
        return tf.estimator.EstimatorSpec(mode, predictions=preds)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        LR = {'start': 5e-3, 'end': 5e-4, 'steps': 1500}
        
        lr_op = tf.train.exponential_decay(
            LR['start'], global_step, LR['steps'], LR['end']/LR['start'])

        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

        train_op = tf.train.AdamOptimizer(lr_op).minimize(
            loss_op, global_step=global_step)

        lth = tf.train.LoggingTensorHook({'lr': lr_op}, every_n_iter=100)
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op, training_hooks=[lth])
