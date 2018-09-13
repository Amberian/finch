import os
import wget     # pip3 install wget
import zipfile


def download(zip_path='../data/WN18/wn18.zip',
             files_path='../data/WN18/'):
    
    if not os.path.isfile('../data/WN18/wn18/train.txt'):
        wget.download(
            'https://github.com/ttrouill/complex/raw/master/datasets/wn18.zip',
            out = zip_path)
        z = zipfile.ZipFile(zip_path, 'r')
        z.extractall(files_path)
        z = z.close()
        os.remove(zip_path)
    else:
        print('Files Already Downloaded')

