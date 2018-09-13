from __future__ import print_function
from data import IMDB
from model import Model
import tensorflow as tf


def main():
    dataloader = IMDB()
    model = Model(dataloader.params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Loading trained model ...")
    model.saver.restore(sess, './saved_temp/model.ckpt')

    # lowercase, no punctuation, please 
    model.post_inference(sess, "i love this film it is one of the best")
    model.post_inference(sess, "this film is awful and the acting is bad")
    model.post_inference(sess, "i hate this boring movie and there is no point to watch")
    

if __name__ == '__main__':
    main()