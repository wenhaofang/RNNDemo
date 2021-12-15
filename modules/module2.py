# LSTM
# Long Short-Term Memory (1997)

import math

import torch
import torch.nn as nn

import collections

class LSTMCell():
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = 1 / hidden_size

        self.W_xi, self.W_hi, self.b_i = self._init_param_group(input_size, hidden_size, k) # input gate
        self.W_xf, self.W_hf, self.b_f = self._init_param_group(input_size, hidden_size, k) # forget gate
        self.W_xo, self.W_ho, self.b_o = self._init_param_group(input_size, hidden_size, k) # output gate
        self.W_xc, self.W_hc, self.b_c = self._init_param_group(input_size, hidden_size, k) # cell

    def _init_param_group(self, input_size, hidden_size, k):
        return (
            self._convert(
                self._uniform(input_size , hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
            ),
            self._convert(
                self._uniform(hidden_size, hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
            ),
            self._convert(
                self._uniform(hidden_size, min_value = -math.sqrt(k), max_value = math.sqrt(k))
            )
        )

    def _uniform(self, *size, min_value, max_value):
        return ((max_value - min_value)* torch.rand(size) + min_value)

    def _convert(self, tensor):
        return nn.Parameter(tensor, requires_grad = True)

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

    def named_parameters(self):
        return nn.ParameterDict({
            'W_xi': self.W_xi, 'W_hi': self.W_hi, 'b_i': self.b_i,
            'W_xf': self.W_xf, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_xo': self.W_xo, 'W_ho': self.W_ho, 'b_o': self.b_o,
            'W_xc': self.W_xc, 'W_hc': self.W_hc, 'b_c': self.b_c
        })

    def state_dict(self):
        return collections.OrderedDict({
            'W_xi': self.W_xi, 'W_hi': self.W_hi, 'b_i': self.b_i,
            'W_xf': self.W_xf, 'W_hf': self.W_hf, 'b_f': self.b_f,
            'W_xo': self.W_xo, 'W_ho': self.W_ho, 'b_o': self.b_o,
            'W_xc': self.W_xc, 'W_hc': self.W_hc, 'b_c': self.b_c
        })

    def load_state_dict(self, state_dict):
        self.W_xi, self.W_hi, self.b_i = (
            self._convert(state_dict['W_xi']), self._convert(state_dict['W_hi']), self._convert(state_dict['b_i'])
        )
        self.W_xf, self.W_hf, self.b_f = (
            self._convert(state_dict['W_xf']), self._convert(state_dict['W_hf']), self._convert(state_dict['b_f'])
        )
        self.W_xo, self.W_ho, self.b_o = (
            self._convert(state_dict['W_xo']), self._convert(state_dict['W_ho']), self._convert(state_dict['b_o'])
        )
        self.W_xc, self.W_hc, self.b_c = (
            self._convert(state_dict['W_xc']), self._convert(state_dict['W_hc']), self._convert(state_dict['b_c'])
        )

    def to(self, device):
        self.W_xi, self.W_hi, self.b_i = (
            self._convert(self.W_xi.to(device)), self._convert(self.W_hi.to(device)), self._convert(self.b_i.to(device))
        )
        self.W_xf, self.W_hf, self.b_f = (
            self._convert(self.W_xf.to(device)), self._convert(self.W_hf.to(device)), self._convert(self.b_f.to(device))
        )
        self.W_xo, self.W_ho, self.b_o = (
            self._convert(self.W_xo.to(device)), self._convert(self.W_ho.to(device)), self._convert(self.b_o.to(device))
        )
        self.W_xc, self.W_hc, self.b_c = (
            self._convert(self.W_xc.to(device)), self._convert(self.W_hc.to(device)), self._convert(self.b_c.to(device))
        )
        return self

if __name__ == '__main__':
    emb_dim = 256
    hid_dim = 512
    batch_size = 64

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    module = LSTMCell(emb_dim, hid_dim).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    X = torch.randn(batch_size, emb_dim).to(device)
    H = torch.zeros(batch_size, hid_dim).to(device)
    C = torch.zeros(batch_size, hid_dim).to(device)

    Y = module(X)
    print(Y[0].shape) # (batch_size, hid_dim)
    print(Y[1].shape) # (batch_size, hid_dim)

    Y = module(X, (H, C))
    print(Y[0].shape) # (batch_size, hid_dim)
    print(Y[1].shape) # (batch_size, hid_dim)
