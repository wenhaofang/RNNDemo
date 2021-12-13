import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic

    parser.add_argument('--name', default = 'main', help = '')

    # For Module

    parser.add_argument('--rnn_type', default = 'rnn', choices = ['rnn', 'lstm', 'gru'], help = '')

    parser.add_argument('--emb_dim', type = int, default = 256, help = '')
    parser.add_argument('--hid_dim', type = int, default = 512, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
