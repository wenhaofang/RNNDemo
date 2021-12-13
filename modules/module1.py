# RNN

import math

import torch
import torch.nn as nn

import collections

class RNNCell():
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1 / hidden_size

        # hidden
        self.W_xh = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hh = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_h  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

    def _uniform(self, *size, min_value, max_value):
        return nn.Parameter(
            (max_value - min_value) * torch.rand(size) + min_value, requires_grad = True
        )

    def __call__(self, X, H = None):
        '''
        Params:
            X: Torch LongTensor (batch_size, emb_dim)
            H: Torch LongTensor (batch_size, hid_dim)
        Return:
            H: Torch LongTensor (batch_size, hid_dim)
        '''
        if H is None:
            H = torch.zeros(X.shape[0], self.hidden_size, device = X.device)

        H = torch.tanh(torch.matmul(X, self.W_xh) + torch.matmul(H, self.W_hh) + self.b_h)

        return H

    def parameters(self):
        return nn.ParameterList([
            self.W_xh, self.W_hh, self.b_h
        ])

    def named_parameters(self):
        return nn.ParameterDict({
            'W_xh': self.W_xh, 'W_hh': self.W_hh, 'b_h': self.b_h
        })

    def state_dict(self):
        return collections.OrderedDict({
            'W_xh': self.W_xh, 'W_hh': self.W_hh, 'b_h': self.b_h
        })

    def load_state_dict(self, state_dict):
        self.W_xh, self.W_hh, self.b_h = state_dict['W_xh'], state_dict['W_hh'], state_dict['b_h']

if __name__ == '__main__':
    emb_dim = 256
    hid_dim = 512
    batch_size = 64

    module = RNNCell(emb_dim, hid_dim)

    X = torch.randn(batch_size, emb_dim)
    H = torch.zeros(batch_size, hid_dim)

    Y = module(X)
    print(Y.shape) # (batch_size, hid_dim)

    Y = module(X, H)
    print(Y.shape) # (batch_size, hid_dim)
