from __future__ import print_function
from data import IMDB
from model import Model
from config import args

import os
import json
import tensorflow as tf


def main():
    dataloader = IMDB()
    model = Model(dataloader.params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_batch = len(dataloader.enc_inp) // args.batch_size
    
    if not os.path.isfile(model.model_path+'.meta'):
        for epoch in range(args.stage_1_num_epochs):
            dataloader.update_word_dropout()
            print("Word Dropout")
            dataloader.shuffle()
            print("Data Shuffled")
            print()
            model.prior_inference(sess)
            model.post_inference(sess, 'i love this film it is so good to watch')
            for i, (enc_inp, dec_inp, dec_out, _) in enumerate(dataloader.next_batch()):
                
                log = model.train_vae_session(sess, enc_inp, dec_inp, dec_out)
                if i % args.stage_1_display_step == 0:
                    print("Step %d | Train VAE | [%d/%d] | [%d/%d]" % (
                        log['step'], epoch+1, args.stage_1_num_epochs, i, n_batch), end='')
                    print(" | nll_loss:%.1f | kl_w:%.2f | kl_loss:%.2f" % (
                        log['nll_loss'], log['kl_w'], log['kl_loss']))
                    print()
        save_path = model.saver.save(sess, model.model_path)
        print("Model saved in file: %s" % save_path)
    else:
        print("Stage 1 Completed")


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()