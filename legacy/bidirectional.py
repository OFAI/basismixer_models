import numpy as np
import torch

from torch import nn

from basismixer.predictive_models import (construct_model as c_model,
                                          NNModel)
from basismixer.predictive_models.architectures import FeedForwardModel, RecurrentModel


class BidirectionalRNN(NNModel):

    def __init__(self, input_size, output_size,
                 recurrent_size, hidden_size,
                 n_layers=1, dropout=0.0,
                 dense_nl=nn.ReLU(),
                 batch_first=False,
                 input_names=None,
                 output_names=None,
                 input_type=None,
                 dtype=torch.float32,
                 device=None):

        super().__init__(input_names=input_names,
                         output_names=output_names,
                         input_type=input_type,
                         dtype=dtype,
                         device=device,
                         is_rnn=True)

        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = True

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.recurrent_size,
                          num_layers=self.n_layers,
                          batch_first=self.batch_first,
                          dropout=self.dropout,
                          bidirectional=True)

        dense_in_features = (self.recurrent_size * (1 + self.rnn.bidirectional))
        self.dense = nn.Linear(in_features=dense_in_features,
                               out_features=self.hidden_size)
        
        self.dense_nl = nn.Identity() if dense_nl is None else dense_nl

        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=self.output_size)

        self._flatten_shape = self.recurrent_size * 2

        if self.batch_first:
            # Index of the batch size
            self._bsi = 0
            # Index of the sequence length
            self._sli = 1
        else:
            # Index of the batch size
            self._bsi = 1
            # Index of the sequence length
            self._sli = 0

    def init_hidden(self, batch_size):

        return torch.zeros(2 * self.n_layers, batch_size, self.recurrent_size)

    def forward(self, input, hprev=None):

        batch_size = input.size(self._bsi)
        seq_len = input.size(self._sli)

        if hprev is None:
            hprev = self.init_hidden(batch_size).type(input.type())

        output, hprev = self.rnn(input, hprev)

        dense = self.dense_nl(self.dense(output.contiguous().view(-1, self._flatten_shape)))

        y = self.out(dense)

        if self.batch_first:
            return y.view(batch_size, seq_len, self.output_size)
        else:
            return y.view(seq_len, batch_size, self.output_size)


if __name__ == '__main__':

    input_size = 7
    output_size = 4
    x = torch.rand(10, 2, input_size)

    input_names = ['i{0}'.format(i) for i in range(input_size)]
    output_names = ['o{0}'.format(o) for o in range(output_size)]
    
    rnn = BidirectionalRNN(input_size=input_size,
                           output_size=output_size,
                           input_names=input_names,
                           output_names=output_names,
                           recurrent_size=8,
                           hidden_size=6, n_layers=1,
                           batch_first=False)
    y = rnn(x)
