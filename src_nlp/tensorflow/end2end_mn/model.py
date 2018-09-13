from config import args

import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    if labels is None:
        labels = tf.zeros([tf.shape(features['inputs'])[0], params['max_answer_len']], tf.int64)

    logits = forward(features, params, is_training=True, seq_inputs=shift_right(labels, params))
    predicted_ids = forward(features, params, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicted_ids)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=logits, targets=labels, weights=tf.ones_like(labels, tf.float32)))

        train_op = tf.train.AdamOptimizer().minimize(loss_op,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)


def hop_forward(features, question, params, is_training, reuse):
    with tf.variable_scope('memory_o', reuse=reuse):
        memory_o = input_mem(features['inputs'], params, is_training)
    
    with tf.variable_scope('memory_i', reuse=reuse):
        memory_i = input_mem(features['inputs'], params, is_training)
    
    match = tf.matmul(question, tf.transpose(memory_i, [0,2,1]))

    match = pre_softmax_masking(match, features['inputs_len'], params['max_input_len'])

    match = tf.nn.softmax(match) # (batch, question_maxlen, input_maxlen)

    match = post_softmax_masking(match, features['questions_len'], params['max_quest_len'])

    response = tf.matmul(match, memory_o)

    with tf.variable_scope('answer', reuse=reuse):
        answer = tf.concat([response, question], -1)
        answer = tf.layers.dense(answer, args.hidden_dim, name='res_quest_proj_ans')
    
    return answer


def forward(features, params, is_training, seq_inputs=None, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('questions', reuse=reuse):
        question = quest_mem(features['questions'], params, is_training)
    
    for _ in range(args['n_hops']):
        answer = hop_forward(features, question, params, is_training, reuse)
        question = answer
    
    with tf.variable_scope('memory_o', reuse=True):
        embedding = tf.get_variable('lookup_table')
    with tf.variable_scope('final_answer', reuse=reuse):
        output = answer_module(features, params, answer, embedding, is_training, seq_inputs)
    
    return output


def input_mem(x, params, is_training):
    x = embed_seq(x, params)
    x = tf.layers.dropout(x, args.dropout_rate, training=is_training)
    pos = position_encoding(params['max_sent_len'], args.hidden_dim)
    x = tf.reduce_sum(x * pos, 2)
    return x


def quest_mem(x, params, is_training):
    x = embed_seq(x, params)
    x = tf.layers.dropout(x, args.dropout_rate, training=is_training)
    pos = position_encoding(params['max_quest_len'], args.hidden_dim)
    return (x * pos)


def answer_module(features, params, answer, embedding, is_training, seq_inputs=None):
    """
    _, answer = tf.nn.dynamic_rnn(
        GRU('answer_3d_to_2d'), answer, tf.count_nonzero(features['questions'], 1), dtype=tf.float32)
    """
    answer = tf.layers.dense(tf.layers.flatten(answer), args.hidden_dim, name='answer_hidden')
    init_state = tf.layers.dropout(answer, args.dropout_rate, training=is_training)

    if is_training:
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = tf.nn.embedding_lookup(embedding, seq_inputs),
            sequence_length = tf.to_int32(features['answers_len']))
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = GRU('decoder_rnn'),
            helper = helper,
            initial_state = init_state,
            output_layer = tf.layers.Dense(params['vocab_size'], name='vocab_proj'))
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder)
        return decoder_output.rnn_output
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding = embedding,
            start_tokens = tf.tile(
                tf.constant([params['<start>']], dtype=tf.int32), [tf.shape(init_state)[0]]),
            end_token = params['<end>'])
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = GRU('decoder_rnn'),
            helper = helper,
            initial_state = init_state,
            output_layer = tf.layers.Dense(params['vocab_size'], name='vocab_proj'))
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            maximum_iterations = params['max_answer_len'])
        return decoder_output.sample_id


def pre_softmax_masking(x, seq_len, max_seq_len):
    paddings = tf.fill(tf.shape(x), float('-inf'))
    T = x.get_shape().as_list()[1]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, T, 1])
    return tf.where(tf.equal(masks, 0), paddings, x)


def post_softmax_masking(x, seq_len, max_seq_len):
    T = x.get_shape().as_list()[-1]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, T])
    return (x * masks)


def shift_right(x, params):
    batch_size = tf.shape(x)[0]
    start = tf.to_int64(tf.fill([batch_size, 1], params['<start>']))
    return tf.concat([start, x[:, :-1]], 1)


def GRU(name, rnn_size=None):
    rnn_size = args.hidden_dim if rnn_size is None else rnn_size
    return tf.nn.rnn_cell.GRUCell(
        rnn_size, kernel_initializer=tf.orthogonal_initializer(), name=name)


def embed_seq(x, params, zero_pad=True):
    lookup_table = tf.get_variable('lookup_table', [params['vocab_size'], args.hidden_dim], tf.float32)
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, args.hidden_dim]), lookup_table[1:, :]), axis=0)
    return tf.nn.embedding_lookup(lookup_table, x)


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, args['clip_norm'])
    return zip(clipped_grads, variables)
