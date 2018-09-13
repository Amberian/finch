from __future__ import print_function
from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, batch_size):
        self.data = {
            'train': {
                'X': {
                    'user_id': None,
                    'gender_id': None,
                    'age_id': None,
                    'job_id': None,
                    'movie_id': None,
                    'category_ids': None,
                    'movie_title': None,
                },
                'Y': None,
            },
            'test': {
                'X': {
                    'user_id': None,
                    'gender_id': None,
                    'age_id': None,
                    'job_id': None,
                    'movie_id': None,
                    'category_ids': None,
                    'movie_title': None,
                },
                'Y': None,
            }
        }
        self.batch_size = batch_size
        self.process('train')
        self.process('test')

    def process(self, routine):
        csv = pd.read_csv('./data/movielens_%s.csv'%routine)

        self.data[routine]['X']['user_id'] = csv['user_id'].values
        self.data[routine]['X']['gender_id'] = csv['gender_id'].values
        self.data[routine]['X']['age_id'] = csv['age_id'].values
        self.data[routine]['X']['job_id'] = csv['job_id'].values
        self.data[routine]['X']['movie_id'] = csv['movie_id'].values

        self.data[routine]['X']['category_ids'] = []
        for category_id in tqdm(
            csv['category_ids'].values, total=len(csv), ncols=70):
            temp_li = [0] * 18
            category_id_li = category_id.split()
            for idx in category_id_li:
                temp_li[int(idx)] = 1
            self.data[routine]['X']['category_ids'].append(temp_li)
        self.data[routine]['X']['category_ids'] = np.array(self.data[routine]['X']['category_ids'])

        self.data[routine]['X']['movie_title'] = []
        for mov_title in tqdm(
            csv['movie_title'].values, total=len(csv), ncols=70):
            temp_li = [0] * 10
            mov_title_li = mov_title.split()
            for i in range(len(mov_title_li[:10])):       
                temp_li[i] = int(mov_title_li[i])
            self.data[routine]['X']['movie_title'].append(temp_li)
        self.data[routine]['X']['movie_title'] = np.array(self.data[routine]['X']['movie_title'])

        self.data[routine]['Y'] = csv['score'].values
    
    def train_pipeline(self):
        return tf.estimator.inputs.numpy_input_fn(
            x = {
                'user_id': self.data['train']['X']['user_id'],
                'gender_id': self.data['train']['X']['gender_id'],
                'age_id': self.data['train']['X']['age_id'],
                'job_id': self.data['train']['X']['job_id'],
                'movie_id': self.data['train']['X']['movie_id'],
                'category_ids': self.data['train']['X']['category_ids'],
                'movie_title': self.data['train']['X']['movie_title']},
            y = self.data['train']['Y'],
            batch_size = self.batch_size,
            shuffle = True)

    def eval_pipeline(self):
        return tf.estimator.inputs.numpy_input_fn(
            x = {
                'user_id': self.data['test']['X']['user_id'],
                'gender_id': self.data['test']['X']['gender_id'],
                'age_id': self.data['test']['X']['age_id'],
                'job_id': self.data['test']['X']['job_id'],
                'movie_id': self.data['test']['X']['movie_id'],
                'category_ids': self.data['test']['X']['category_ids'],
                'movie_title': self.data['test']['X']['movie_title']},
            batch_size = self.batch_size,
            y = self.data['test']['Y'],
            shuffle = False)
