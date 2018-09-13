import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_val_split', type=float, default=0.95)
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
