from tqdm import tqdm
from configs.general import args

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class PreProcessor:
    def __init__(self):
        self.csv_train_path = '../data/files_original/train.csv'
        self.csv_test_path = '../data/files_original/test.csv'
        self.question_path = '../data/files_original/question.csv'
        self.embedding_path = '../data/files_original/char_embed.txt'

        self.cleaned_question_path = '../data/files_processed/question_cleaned.csv'
        self.save_embedding_path = '../data/files_processed/char_embedding.npy'
        self.q2c_path = '../data/files_processed/q2c.pkl'
        
        self.tfrecord_train_path = '../data/tfrecords/char_train.tfrecord'
        self.tfrecord_val_path = '../data/tfrecords/char_val.tfrecord'
        self.tfrecord_test_path = '../data/tfrecords/char_test.tfrecord'

        self.clean_text()
        self.load_embedding()
        self.make_qid_to_word_dict()
        self.make_tfrecords()
    

    def clean_text(self):
        if os.path.isfile(self.cleaned_question_path):
            tf.logging.info("Exists %s" % self.cleaned_question_path)
        else:
            if not os.path.exists(os.path.dirname(self.cleaned_question_path)):
                os.makedirs(os.path.dirname(self.cleaned_question_path))
            with open(self.question_path) as f:
                st = f.read()
                st = st.replace('W0', '')
                st = st.replace('L0', '')
                st = st.replace('W', '')
                st = st.replace('L', '')
            with open(self.cleaned_question_path, 'w') as f:
                f.write(st)

    def load_embedding(self):
        if os.path.isfile(self.save_embedding_path):
            tf.logging.info("Exists %s" % self.save_embedding_path)
        else:
            if not os.path.exists(os.path.dirname(self.save_embedding_path)):
                os.makedirs(os.path.dirname(self.save_embedding_path))
            embed_vals = []
            with open(self.embedding_path) as f:
                for line in f:
                    line_sp = line.split()
                    embed_vals.append([float(num) for num in line_sp[1:]])
                embed_vals = np.asarray(embed_vals, dtype=np.float32)
                PAD_INT = embed_vals.shape[0]
                zeros = np.zeros((1,300), dtype=np.float32)
                embed_vals = np.concatenate([embed_vals, zeros])
                np.save(self.save_embedding_path, embed_vals)
    
    def make_qid_to_word_dict(self):
        if os.path.isfile(self.q2c_path):
            tf.logging.info("Exists %s" % self.q2c_path)
        else:
            if not os.path.exists(os.path.dirname(self.q2c_path)):
                os.makedirs(os.path.dirname(self.q2c_path))
        
            def fn(path):
                _q2w, _q2c = {}, {}
                _w_lens, _c_lens = [], []

                with open(path) as f:
                    next(f)
                    for line in f:
                        l_split = line.split(',')
                        qid, words, chars = l_split

                        words_sp = words.split()
                        chars_sp = chars.split()

                        _q2w[qid] = words_sp
                        _q2c[qid] = chars_sp

                        _w_lens.append(len(words_sp))
                        _c_lens.append(len(chars_sp))
                        
                return _q2w, _q2c, _w_lens, _c_lens

            q2w, q2c, w_lens, c_lens = fn(self.cleaned_question_path)
            save_obj(q2c, self.q2c_path)
    

    def make_tfrecords(self):
        if (os.path.isfile(self.tfrecord_train_path)) and (os.path.isfile(self.tfrecord_val_path) and (os.path.isfile(self.tfrecord_test_path))):
            tf.logging.info("Exists %s" % self.tfrecord_train_path)
            tf.logging.info("Exists %s" % self.tfrecord_val_path)
            tf.logging.info("Exists %s" % self.tfrecord_test_path)
        else:
            if not os.path.exists(os.path.dirname(self.tfrecord_train_path)):
                os.makedirs(os.path.dirname(self.tfrecord_train_path))
            q2c = load_obj(self.q2c_path)

            ori_train_csv = pd.read_csv(self.csv_train_path)
            thres = int(len(ori_train_csv) * args.train_val_split)
            train_csv = ori_train_csv[:thres]
            val_csv = ori_train_csv[thres:]
            test_csv = pd.read_csv(self.csv_test_path)

            embed_vals = np.load(self.save_embedding_path)
            PAD_INT = len(embed_vals) - 1

            train_fn(train_csv, self.tfrecord_train_path, PAD_INT, q2c)
            train_fn(val_csv, self.tfrecord_val_path, PAD_INT, q2c)
            test_fn(test_csv, self.tfrecord_test_path, PAD_INT, q2c)


def train_fn(csv, path, PAD_INT, q2c):
    writer = tf.python_io.TFRecordWriter(path)
    for arr_line in tqdm(csv.values, total=len(csv), ncols=70):
        label, q1_id, q2_id = arr_line

        q1_id_int = [int(st) for st in q2c[q1_id]]
        q2_id_int = [int(st) for st in q2c[q2_id]]
        
        example = tf.train.Example(
            features = tf.train.Features(
                 feature = {
                   'input1': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=q1_id_int)),
                   'input2': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=q2_id_int)),
                   'label': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=[label])),
                   }))
        serialized = example.SerializeToString()
        writer.write(serialized)


def test_fn(csv, path, PAD_INT, q2c):
    writer = tf.python_io.TFRecordWriter(path)
    for arr_line in tqdm(csv.values, total=len(csv), ncols=70):
        q1_id, q2_id = arr_line

        q1_id_int = [int(st) for st in q2c[q1_id]]
        q2_id_int = [int(st) for st in q2c[q2_id]]
        
        example = tf.train.Example(
            features = tf.train.Features(
                 feature = {
                   'input1': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=q1_id_int)),
                   'input2': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=q2_id_int)),
                   }))
        serialized = example.SerializeToString()
        writer.write(serialized)

