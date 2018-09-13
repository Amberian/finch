from config import args
from attn_gru_cell import AttentionGRUCell

import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    if labels is None:
        labels = tf.placeholder(tf.int64, [None, params['max_answer_len']])

    logits = forward(features, params, is_training=True, seq_inputs=shift_right(labels, params))
        
    predicted_ids = forward(features, params, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicted_ids)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=logits, targets=labels, weights=tf.ones_like(labels, tf.float32)))

        variables = tf.trainable_variables()
        grads = tf.gradients(loss_op, variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)

        train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_grads, variables),
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)


def forward(features, params, is_training, seq_inputs=None, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('word_embedding', reuse=reuse):
        embedding = tf.get_variable('lookup_table', [params['vocab_size'], args.embed_dim], tf.float32)
        embedding = zero_index_pad(embedding)

    with tf.variable_scope('input_module', reuse=reuse):
        fact_vecs = input_module(features, params, embedding, is_training)
    
    with tf.variable_scope('question_module', reuse=reuse):
        q_vec = question_module(features, embedding)
    
    with tf.variable_scope('question_module', reuse=reuse):
        memory = memory_module(features, fact_vecs, q_vec, is_training)

    with tf.variable_scope('answer_module', reuse=reuse):
        logits = answer_module(features, params, memory, q_vec, embedding, is_training, seq_inputs)
    
    return logits


def input_module(features, params, embedding, is_training):
    inputs = tf.nn.embedding_lookup(embedding, features['inputs'])         # (B, I, S, D)
    position = position_encoding(params['max_sent_len'], args.embed_dim)
    inputs = tf.reduce_sum(inputs * position, 2)                           # (B, I, D)
    birnn_out, _ = tf.nn.bidirectional_dynamic_rnn(                                             
        GRU('input_birnn_fw', args.hidden_size//2),
        GRU('input_birnn_bw', args.hidden_size//2),
        inputs, features['inputs_len'], dtype=np.float32)
    fact_vecs = tf.concat(birnn_out, -1)                                   # (B, I, D)
    fact_vecs = tf.layers.dropout(fact_vecs, args.dropout_rate, training=is_training)
    return fact_vecs


def question_module(features, embedding):
    questions = tf.nn.embedding_lookup(embedding, features['questions'])
    _, q_vec = tf.nn.dynamic_rnn(
        GRU('question_rnn'), questions, features['questions_len'], dtype=np.float32)
    return q_vec


def memory_module(features, fact_vecs, q_vec, is_training):
    memory = q_vec
    for i in range(args.n_hops):
        print('==> Memory Episode', i)
        episode = gen_episode(features, memory, q_vec, fact_vecs, is_training)
        memory = tf.layers.dense(
            tf.concat([memory, episode, q_vec], 1), args.hidden_size, tf.nn.relu, name='memory_proj')
    return memory  # (B, D)


def gen_episode(features, memory, q_vec, fact_vecs, is_training):
    def gen_attn(fact_vec):
        features = [fact_vec * q_vec,
                    fact_vec * memory,
                    tf.abs(fact_vec - q_vec),
                    tf.abs(fact_vec - memory)]
        feature_vec = tf.concat(features, 1)
        attention = tf.layers.dense(feature_vec, args.embed_dim, tf.tanh, name='attn_proj_1')
        attention = tf.layers.dense(attention, 1, name='attn_proj_2')
        return tf.squeeze(attention, 1)

    # Gates (attentions) are activated, if sentence relevant to the question or memory
    attns = tf.map_fn(gen_attn, tf.transpose(fact_vecs, [1,0,2]))
    attns = tf.transpose(attns)                                    # (B, n_fact)
    attns = tf.nn.softmax(attns)                                   # (B, n_fact)
    attns = tf.expand_dims(attns, -1)                              # (B, n_fact, 1)
    
    # The relevant facts are summarized in another GRU
    _, episode = tf.nn.dynamic_rnn(
        AttentionGRUCell(args.hidden_size, name='attn_gru'),
        tf.concat([fact_vecs, attns], 2),                          # (B, n_fact, D+1)
        features['inputs_len'],
        dtype=np.float32)
    return episode                                                 # (B, D)


def answer_module(features, params, memory, q_vec, embedding, is_training, seq_inputs=None):
    memory = tf.layers.dropout(memory, args.dropout_rate, training=is_training)
    init_state = tf.layers.dense(tf.concat((memory, q_vec), -1), args.hidden_size, name='answer_proj')
    
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


def shift_right(x, params):
    batch_size = tf.shape(x)[0]
    start = tf.to_int64(tf.fill([batch_size, 1], params['<start>']))
    return tf.concat([start, x[:, :-1]], 1)


def GRU(name, rnn_size=None):
    rnn_size = args.hidden_size if rnn_size is None else rnn_size
    return tf.nn.rnn_cell.GRUCell(
        rnn_size, kernel_initializer=tf.orthogonal_initializer(), name=name)


def zero_index_pad(embedding):
    return tf.concat((tf.zeros([1, args.embed_dim]), embedding[1:, :]), axis=0)


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)
