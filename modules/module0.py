import math

import torch
import torch.nn as nn

from modules.module1 import RNNCell
from modules.module2 import LSTMCell
from modules.module3 import GRUCell

class Linear():
    def __init__(self, in_features, out_features):

        k = 1 / in_features

        self.W = self._uniform(in_features , out_features, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b = self._uniform(out_features, min_value = -math.sqrt(k), max_value = math.sqrt(k))

    def _uniform(self, *size, min_value, max_value):
        return nn.Parameter(
            (max_value - min_value) * torch.rand(size) + min_value, requires_grad = True
        )

    def __call__(self, X):
        '''
        Params:
            inputs : Torch LongTensor (batch_size, in_features )
        Return:
            outputs: Torch LongTensor (batch_size, out_features)
        '''
        Y = torch.matmul(X, self.W) + self.b
        return Y

    def parameters(self):
        return nn.ParameterList([self.W, self.b])

class LanguageModel():
    def __init__(self, emb_dim, hid_dim, vocab_size, rnn_type):

        assert rnn_type.lower() in ['rnn', 'lstm', 'gru']

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type

        if rnn_type.lower() == 'rnn' :
            self.r_cell = RNNCell (emb_dim, hid_dim)
        if rnn_type.lower() == 'lstm':
            self.r_cell = LSTMCell(emb_dim, hid_dim)
        if rnn_type.lower() == 'gru' :
            self.r_cell = GRUCell (emb_dim, hid_dim)

        self.linear = Linear(hid_dim, vocab_size)

    def parameters(self):
        result = nn.ParameterList()
        result.extend(self.r_cell.parameters())
        result.extend(self.linear.parameters())
        return result

    def __call__(self, inputs, hidden = None):
        '''
        Params:
            inputs: Torch LongTensor (batch_size, seq_len, emb_dim)
            hidden: Torch LongTensor (batch_size, hid_dim) if rnn_type == 'rnn' or rnn_type == 'gru'
                    (
                        Torch LongTensor (batch_size, hid_dim),
                        Torch LongTensor (batch_size, hid_dim)
                    ) if rnn_type == 'lstm'
        Return:
            outputs: Torch LongTensor (batch_size, seq_len, vocab_size)
        '''
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device = inputs.device)
        # todo: teacher forcing
        for t in range(seq_len):
            hidden = self.r_cell(inputs[:, t, :], hidden)
            output = self.linear(hidden if self.rnn_type == 'rnn' or self.rnn_type == 'gru' else hidden[0])
            outputs[:, t, :] = output
        return outputs

def get_module(option, vocab_size):
    return LanguageModel(
        vocab_size, # option.emb_dim
        option.hid_dim,
        vocab_size,
        option.rnn_type
    )

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    vocab_size = 1000

    module = get_module(option, vocab_size)

    batch_size = 128
    max_seq_len = 70

    X = torch.randn(batch_size, max_seq_len, vocab_size)
    Y = module(X) #(batch_size, max_seq_len, vocab_size)
