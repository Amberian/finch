import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=10)

parser.add_argument('--vocab_size', type=int, default=20000)
parser.add_argument('--num_sampled', type=int, default=1000)
parser.add_argument('--word_dropout_rate', type=float, default=0.8)

parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--embed_dims', type=int, default=128)

parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--clip_norm', type=float, default=5.0)

parser.add_argument('--vae_ckpt_dir', type=str, default='../model/ckpt/vae')
parser.add_argument('--vae_display_step', type=int, default=100)
parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--anneal_max', type=float, default=1.0)
parser.add_argument('--anneal_bias', type=int, default=6000)

parser.add_argument('--discriminator_dropout_rate', type=float, default=0.2)
parser.add_argument('--n_filters', type=int, default=128)
parser.add_argument('--n_class', type=int, default=2)

parser.add_argument('--wake_sleep_display_step', type=int, default=1)
parser.add_argument('--temp_anneal_max', type=float, default=1.0)
parser.add_argument('--temp_anneal_bias', type=int, default=1000)
parser.add_argument('--lambda_c', type=float, default=0.1)
parser.add_argument('--lambda_z', type=float, default=0.1)
parser.add_argument('--lambda_u', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.1)

args = parser.parse_args()
