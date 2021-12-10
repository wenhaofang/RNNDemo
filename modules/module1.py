# RNN

import math

import torch
import torch.nn as nn

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

        return (H,)

if __name__ == '__main__':
    emb_dim = 256
    hid_dim = 512
    batch_size = 64

    module = RNNCell(emb_dim, hid_dim)

    X = torch.randn(batch_size, emb_dim)
    H = torch.zeros(batch_size, hid_dim)

    Y = module(X)
    print(Y[0].shape) # (batch_size, hid_dim)

    Y = module(X, H)
    print(Y[0].shape) # (batch_size, hid_dim)
