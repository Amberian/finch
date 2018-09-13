import tensorflow as tf

from configs import args


class IMDBVocab(object):
    def __init__(self):
        index_from = 4
        self.word2idx = tf.keras.datasets.imdb.get_word_index()
        self.word2idx = {k: (v + index_from) for k, v in self.word2idx.items() if v <= args.vocab_size-index_from}
        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<unk>'] = 2
        self.word2idx['<end>'] = 3
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
