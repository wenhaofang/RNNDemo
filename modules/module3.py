# GRU [2014]
# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

import math

import torch
import torch.nn as nn

class GRUCell():
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1 / hidden_size

        # update gate
        self.W_xz = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hz = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_z  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

        # reset gate
        self.W_xr = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hr = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_r  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

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

        Z = torch.sigmoid(torch.matmul(X, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z)
        R = torch.sigmoid(torch.matmul(X, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r)
        H_tilda = torch.tanh(torch.matmul(X, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilda

        return (H,)

if __name__ == '__main__':
    emb_dim = 256
    hid_dim = 512
    batch_size = 64

    module = GRUCell(emb_dim, hid_dim)

    X = torch.randn(batch_size, emb_dim)
    H = torch.zeros(batch_size, hid_dim)

    Y = module(X)
    print(Y[0].shape) # (batch_size, hid_dim)

    Y = module(X, H)
    print(Y[0].shape) # (batch_size, hid_dim)
