import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic

    parser.add_argument('--name', default = 'main', help = '')

    # For Loader

    parser.add_argument('--data_path', default = 'data/jaychou_lyrics.txt', help = '')

    parser.add_argument('--min_freq', type = int, default = 1, help = '')
    parser.add_argument('--max_numb', type = int, default = 10000, help = '')

    parser.add_argument('--max_seq_len', type = int, default = 70, help = '')

    # For Module

    parser.add_argument('--rnn_type', default = 'rnn', choices = ['rnn', 'lstm', 'gru'], help = '')

    parser.add_argument('--emb_dim', type = int, default = 256, help = '')
    parser.add_argument('--hid_dim', type = int, default = 512, help = '')

    # For Train

    parser.add_argument('--batch_size', type = int, default = 128, help = '')
    parser.add_argument('--num_epochs', type = int, default = 100, help = '')

    parser.add_argument('--learning_rate', type = float, default = 1e2, help = '')
    parser.add_argument('--clipping_hold', type = float, default = 1e-2, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
