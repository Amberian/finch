from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import sklearn


class Conv1DClassifier:
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(),
                 n_filters=250, embedding_dims=50, padding='valid', top_k=5):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        embedding_dims: int
            Word embedding dimensions
        n_filters: int
            Number output of filters in the convolution
        kernel_size: int
            Size of the 1D convolution window
        hidden_dims: int
            Ouput dimensions of the fully-connected layer
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object 
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_filters = n_filters
        self.embedding_dims = embedding_dims
        self.padding = padding
        self.n_out = n_out
        self.top_k = top_k
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()

        self.kernels = [3, 4, 5]
        parallels = []
        for k in self.kernels:
            p = self.add_conv1d(self.n_filters//len(self.kernels), kernel_size=k)
            p = self.add_kmax_pooling(p)
            parallels.append(p)
        self.merge_layers(parallels)

        self.add_fully_connected_layer()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
        self._batch_size = tf.shape(self.X)[0]
    # end method add_input_layer


    def add_word_embedding(self):
        embedding = tf.get_variable('encoder', [self.vocab_size,self.embedding_dims], tf.float32)
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def add_conv1d(self, n_filters, kernel_size, strides=1):
        Y = tf.layers.conv1d(inputs = self._pointer,
                             filters = n_filters,
                             kernel_size  = kernel_size,
                             strides = strides,
                             padding = self.padding,
                             use_bias = True,
                             activation = tf.nn.relu)
        return Y
    # end method add_conv1d_layer


    def add_kmax_pooling(self, x):
        Y = tf.transpose(x, [0, 2, 1])
        Y = tf.nn.top_k(Y, self.top_k, sorted=False).values
        Y = tf.transpose(Y, [0, 2, 1])
        Y = tf.reshape(Y, [self._batch_size, self.top_k, self.n_filters//len(self.kernels)])
        return Y
    # end method add_kmax_pooling


    def merge_layers(self, layers):
        self._pointer = tf.concat(layers, axis=-1)
    # end method merge_layers


    def add_fully_connected_layer(self):
        self._pointer = tf.reshape(self._pointer,
            [self._batch_size, self.top_k * (len(self.kernels)*(self.n_filters//len(self.kernels)))])
        self._pointer = tf.layers.dense(self._pointer, self.n_filters, tf.nn.relu)
        self._pointer = tf.nn.dropout(self._pointer, self.keep_prob)
    # end method add_flatten_layer
    

    def add_output_layer(self):
        self.logits = tf.layers.dense(self._pointer, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), tf.float32))
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=0.8, en_exp_decay=True,
            en_shuffle=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            local_step = 1
            
            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size),
                                        self.gen_batch(Y, batch_size)): # batch training
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X:X_batch, self.Y:Y_batch,
                                              self.lr:lr, self.keep_prob:keep_prob})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X:X_test_batch, self.Y:Y_test_batch,
                                                   self.keep_prob:1.0})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)
            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                    "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, {self.X:X_test_batch, self.keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.005
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class