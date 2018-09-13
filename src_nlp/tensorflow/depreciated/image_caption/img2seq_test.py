from __future__ import print_function
from data_loader import get_loader, load_image 
from build_vocab import Vocabulary
from torchvision import transforms
from PIL import Image
from img2seq import Image2Seq
import os
import pickle
import numpy as np
import tensorflow as tf

    
def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args['vocab_path'], 'rb') as f:
        vocab = pickle.load(f) 

    # Build data loader
    data_loader = get_loader(args['image_dir'], args['caption_path'], vocab, 
                             transform, args['batch_size'],
                             shuffle=True, num_workers=args['num_workers']) 

    # Train the model
    model = Image2Seq((64, 64), vocab.word2idx)
    model.sess.run(tf.global_variables_initializer())
    print("Model Compiled")
    for epoch in range(args['n_epoch']):
        for i, (images, captions, lengths) in enumerate(data_loader):
            loss = model.partial_fit(images.numpy(), captions.numpy(), lengths)
            print('[%d / %d] Loss: %.4f' % (i, len(data_loader), loss))
            if i % 20 == 0:
                sample_image = load_image(args['sample_img']).numpy()
                model.infer(sample_image, vocab.idx2word)

                
if __name__ == '__main__':
    args = {
        'vocab_path': './data/vocab.pkl',
        'image_dir': './data/resized2014',
        'caption_path': './data/annotations/captions_train2014.json',
        'num_workers': 2,
        'n_epoch': 1,
        'batch_size': 128,
        'sample_img': './data/example.png',
    }
    main(args)
