import tensorflow as tf

from configs.general import args
from .preprocess_word_fixed import PreProcessor


def _parse_train_fn(example_proto):
            parsed_feats = tf.parse_single_example(
                example_proto,
                features={
                    'input1': tf.FixedLenFeature([args.w_max_len], tf.int64),
                    'input2': tf.FixedLenFeature([args.w_max_len], tf.int64),
                    'label': tf.FixedLenFeature([], tf.int64)
                })
            return parsed_feats['input1'], parsed_feats['input2'], parsed_feats['label']


def _parse_test_fn(example_proto):
            parsed_feats = tf.parse_single_example(
                example_proto,
                features={
                    'input1': tf.FixedLenFeature([args.w_max_len], tf.int64),
                    'input2': tf.FixedLenFeature([args.w_max_len], tf.int64),
                })
            return parsed_feats['input1'], parsed_feats['input2']


class DataLoader(object):
    def __init__(self):
        preprocessor = PreProcessor()


    def train_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/word_train_fixed.tfrecord'])
        ds = ds.map(_parse_train_fn)
        ds = ds.shuffle(250000)
        ds = ds.batch(args.batch_size)
        iterator = ds.make_one_shot_iterator()
        x1, x2, y = iterator.get_next()
        return ({'input1': x1, 'input2': x2}, y)


    def val_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/word_val_fixed.tfrecord'])
        ds = ds.map(_parse_train_fn)
        ds = ds.batch(args.batch_size)
        iterator = ds.make_one_shot_iterator()
        x1, x2, y = iterator.get_next()
        return ({'input1': x1, 'input2': x2})

    
    def predict_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/word_test_fixed.tfrecord'])
        ds = ds.map(_parse_test_fn)
        ds = ds.batch(args.batch_size)
        iterator = ds.make_one_shot_iterator()
        x1, x2 = iterator.get_next()
        return ({'input1': x1, 'input2': x2})
