import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--vocab_size', type=int, default=5000)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--embed_dim', type=int, default=50)
parser.add_argument('--filters', type=int, default=250)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=2)

args = parser.parse_args()