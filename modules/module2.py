# LSTM
# Long Short-Term Memory (1997)

import math

import torch
import torch.nn as nn

class LSTMCell():
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1 / hidden_size

        # input gate
        self.W_xi = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hi = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_i  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

        # forget gate
        self.W_xf = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hf = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_f  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

        # output gate
        self.W_xo = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_ho = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_o  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

        # cell
        self.W_xc = self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.W_hc = self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
        self.b_c  = self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))

    def _uniform(self, *size, min_value, max_value):
        return nn.Parameter(
            (max_value - min_value) * torch.rand(size) + min_value, requires_grad = True
        )

    def __call__(self, X, T = None):
        '''
        Params:
            X: Torch LongTensor (batch_size, emb_dim)
            T: Tuple (
                H: Torch LongTensor (batch_size, hid_dim),
                C: Torch LongTensor (batch_size, hid_dim)
            )
        Return:
            T: Tuple (
                H: Torch LongTensor (batch_size, hid_dim),
                C: Torch LongTensor (batch_size, hid_dim)
            )
        '''
        if T is None:
            H = torch.zeros(X.shape[0], self.hidden_size, device = X.device)
            C = torch.zeros(X.shape[0], self.hidden_size, device = X.device)
        else:
            H = T[0]
            C = T[1]

        I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
        F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
        O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
        C_tilda = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()

        return (H, C)

    def parameters(self):
        return nn.ParameterList([
            self.W_xi, self.W_hi, self.b_i,
            self.W_xf, self.W_hf, self.b_f,
            self.W_xo, self.W_ho, self.b_o,
            self.W_xc, self.W_hc, self.b_c
        ])

if __name__ == '__main__':
    emb_dim = 256
    hid_dim = 512
    batch_size = 64

    module = LSTMCell(emb_dim, hid_dim)

    X = torch.randn(batch_size, emb_dim)
    H = torch.zeros(batch_size, hid_dim)
    C = torch.zeros(batch_size, hid_dim)

    Y = module(X)
    print(Y[0].shape) # (batch_size, hid_dim)
    print(Y[1].shape) # (batch_size, hid_dim)

    Y = module(X, (H, C))
    print(Y[0].shape) # (batch_size, hid_dim)
    print(Y[1].shape) # (batch_size, hid_dim)
