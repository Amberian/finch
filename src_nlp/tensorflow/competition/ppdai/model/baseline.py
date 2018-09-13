from configs.general import args

import tensorflow as tf
import numpy as np
import time
import pprint


def load_embedding():
    t0 = time.time()
    embedding = np.load('../data/files_processed/word_embedding.npy')
    print("Load word_embed: %.2fs"%(time.time()-t0))
    return embedding


def mask_fn(x):
    return tf.sign(tf.reduce_sum(x, -1))


def embed_dropout(x, embedding, is_training):
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.layers.dropout(x, 0.2, is_training)
    return x


def birnn(x, cell_fw, cell_bw):
    seq_len = tf.count_nonzero(tf.reduce_sum(x, -1), 1)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                      cell_bw,
                                                      x,
                                                      seq_len,
                                                      dtype=tf.float32)
    outputs = tf.concat(outputs, -1)
    return outputs


def attn_pool(x, proj, alpha, masks):
    x = proj(x)
    align = tf.reduce_sum(alpha * tf.tanh(x), axis=-1)
    # masking
    paddings = tf.fill(tf.shape(align), float('-inf'))
    align = tf.where(tf.equal(masks, 0), paddings, align)
    # probability
    align = tf.expand_dims(tf.nn.softmax(align), -1)
    # weighted sum
    x = tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)
    return x


def query_context_attn(query, context, v, w_k, w_v, masks):
    query = tf.expand_dims(query, 1)
    keys = w_k(context)
    values = w_v(context)

    align = v * tf.tanh(query + keys)
    align = tf.reduce_sum(align, 2)

    paddings = tf.fill(tf.shape(align), float('-inf'))
    align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    align = tf.expand_dims(align, -1)
    val = tf.squeeze(tf.matmul(values, align, transpose_a=True), -1)
    return val


def forward(features, mode):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    x1, x2 = features['input1'], features['input2']
    hidden_units = 100

    with tf.variable_scope('representation'):
        embedding = tf.convert_to_tensor(load_embedding())
        x1 = embed_dropout(x1, embedding, is_training)
        x2 = embed_dropout(x2, embedding, is_training)
        mask1 = mask_fn(x1)
        mask2 = mask_fn(x2)

        cell_fw = tf.nn.rnn_cell.LSTMCell(150, initializer=tf.orthogonal_initializer())
        cell_bw = tf.nn.rnn_cell.LSTMCell(150, initializer=tf.orthogonal_initializer())
        x1 = birnn(x1, cell_fw, cell_bw)
        x2 = birnn(x2, cell_fw, cell_bw)

    with tf.variable_scope('attention_pooling'):
        proj = tf.layers.Dense(hidden_units)
        alpha = tf.get_variable('alpha', [hidden_units])
        attn_pool_1 = attn_pool(x1, proj, alpha, mask1)
        attn_pool_2 = attn_pool(x2, proj, alpha, mask2)

    with tf.variable_scope('query_context_attention'):
        v = tf.get_variable('v', [hidden_units])
        proj_k = tf.layers.Dense(hidden_units)
        proj_v = tf.layers.Dense(hidden_units)
        query_context_attn_1 = query_context_attn(attn_pool_1, x2, v, proj_k, proj_v, mask2)
        query_context_attn_2 = query_context_attn(attn_pool_2, x1, v, proj_k, proj_v, mask1)
    
    with tf.variable_scope('aggregation'):
        feat1 = attn_pool_1
        feat2 = attn_pool_2
        feat3 = tf.abs(feat1 - feat2)
        feat4 = feat1 * feat2
        feat5 = query_context_attn_1
        feat6 = query_context_attn_2
        feat7 = tf.abs(query_context_attn_1 - query_context_attn_2)
        feat8 = query_context_attn_1 * query_context_attn_2
        m1 = tf.reduce_max(x1, 1)
        m2 = tf.reduce_max(x2, 1)
        feat9 = tf.abs(m1 - m2)
        feat10 = m1 * m2

        x = tf.concat([feat1,
                       feat2,
                       feat3,
                       feat4,
                       feat5,
                       feat6,
                       feat7,
                       feat8,
                       feat9,
                       feat10], -1)
        
        x = tf.layers.dropout(x, 0.5, training=is_training)
        x = tf.layers.dense(x, 100, tf.nn.elu)
        x = tf.layers.dropout(x, 0.2, training=is_training)
        x = tf.layers.dense(x, 20, tf.nn.elu)

        x = tf.squeeze(tf.layers.dense(x, 1), -1)
    
    return x


def clip_grads(loss):
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
    return zip(clipped_grads, params)


def model_fn(features, labels, mode):
    logits = forward(features, mode)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.sigmoid(logits))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
        global_step = tf.train.get_global_step()

        LR = {'start': 1e-3, 'end': 5e-4, 'steps': 10000}
        
        lr_op = tf.train.exponential_decay(
            LR['start'], global_step, LR['steps'], LR['end']/LR['start'])
        
        nll_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(labels)))
        
        """
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ])
        l2_loss *= 1e-4
        """
        
        loss_op = nll_loss
        clipped = clip_grads(loss_op)
        train_op = tf.train.AdamOptimizer(lr_op).apply_gradients(
            clipped, global_step=global_step)

        lth = tf.train.LoggingTensorHook({'lr': lr_op}, every_n_iter=100)
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op, training_hooks=[lth])
