import tensorflow as tf

from configs.general import args
from .preprocess_char_rnn import PreProcessor


def _parse_train_fn(example_proto):
            parsed_feats = tf.parse_single_example(
                example_proto,
                features={
                    'input1': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    'input2': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    'label': tf.FixedLenFeature([], tf.int64)
                })
            return parsed_feats


def _parse_test_fn(example_proto):
            parsed_feats = tf.parse_single_example(
                example_proto,
                features={
                    'input1': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    'input2': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                })
            return parsed_feats


class DataLoader(object):
    def __init__(self):
        preprocessor = PreProcessor()


    def train_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/char_train.tfrecord'])
        ds = ds.map(_parse_train_fn)
        ds = ds.shuffle(250000)
        ds = ds.padded_batch(args.batch_size, {'input1': [None], 'input2': [None], 'label': []})
        iterator = ds.make_one_shot_iterator()
        batch = iterator.get_next()
        x1, x2, y = batch['input1'], batch['input2'], batch['label']
        return ({'input1': x1, 'input2': x2}, y)


    def val_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/char_val.tfrecord'])
        ds = ds.map(_parse_train_fn)
        ds = ds.padded_batch(args.batch_size, {'input1': [None], 'input2': [None], 'label': []})
        iterator = ds.make_one_shot_iterator()
        batch = iterator.get_next()
        x1, x2 = batch['input1'], batch['input2']
        return ({'input1': x1, 'input2': x2})

    
    def predict_input_fn(self):
        ds = tf.data.TFRecordDataset(['../data/tfrecords/char_test.tfrecord'])
        ds = ds.map(_parse_test_fn)
        ds = ds.padded_batch(args.batch_size, {'input1': [None], 'input2': [None]})
        iterator = ds.make_one_shot_iterator()
        batch = iterator.get_next()
        x1, x2 = batch['input1'], batch['input2']
        return ({'input1': x1, 'input2': x2})
