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
    
    print("Loading trained model ...")
    model.saver.restore(sess, model.model_path)

    for epoch in range(args.stage_2_num_epochs):
        dataloader.update_word_dropout()
        print("Word Dropout")
        dataloader.shuffle()
        print("Data Shuffled")
        print()
        for i, (enc_inp, dec_inp, dec_out, labels) in enumerate(dataloader.next_batch()):

            log = model.train_discriminator_session(sess, enc_inp, dec_inp, dec_out, labels)
            if i % args.stage_2_display_step == 0:
                print("------------")
                print("Step %d | Train Discriminator | [%d/%d] | [%d/%d]" % (
                    log['step'], epoch+1, args.stage_2_num_epochs, i, n_batch))
                print("\t| clf_loss:%.2f | clf_acc:%.2f | L_u: %.2f" % (
                    log['clf_loss'], log['clf_acc'], log['L_u']))
                print()
            
            log = model.train_encoder_session(sess, enc_inp, dec_inp, dec_out)
            if i % args.stage_2_display_step == 0:
                print("Step %d | Train Encoder | [%d/%d] | [%d/%d]" % (
                    log['step'], epoch+1, args.stage_2_num_epochs, i, n_batch))
                print("\t| seq_loss:%.1f | kl_w:%.2f | kl_loss:%.2f" % (
                    log['nll_loss'], log['kl_w'], log['kl_loss']))
                print()
            
            log = model.train_generator_session(sess, enc_inp, dec_inp, dec_out)
            if i % args.stage_2_display_step == 0:
                print("Step %d | Train Generator | [%d/%d] | [%d/%d]" % (
                    log['step'], epoch+1, args.stage_2_num_epochs, i, n_batch))
                print("\t| seq_loss:%.1f | kl_w:%.2f | kl_loss:%.2f" % (
                    log['nll_loss'], log['kl_w'], log['kl_loss']))
                print("\t| temperature:%.2f | l_attr_z:%.2f | l_attr_c:%.2f" % (
                    log['temperature'], log['l_attr_z'], log['l_attr_c']))
                print("------------")
            
            if i % (5 * args.stage_2_display_step) == 0:
                model.post_inference(sess, 'i love this film it is so good to watch')
                model.post_inference(sess, 'this movie is horrible and waste my time')
        save_path = model.saver.save(sess, './saved_temp/model.ckpt')
        print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()