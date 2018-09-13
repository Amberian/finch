import os
import wget     # pip3 install wget
import zipfile


def download(zip_path='../data/FB15k_237/FB15K-237.2.zip',
             files_path='../data/FB15k_237/'):
    
    if not os.path.isfile('../data/FB15k_237/Release/train.txt'):
        wget.download(
            'https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip',
            out = zip_path)
        z = zipfile.ZipFile(zip_path, 'r')
        z.extractall(files_path)
        z = z.close()
        os.remove(zip_path)
    else:
        print('Files Already Downloaded')

