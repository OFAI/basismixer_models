import math

import numpy as np
import torch

from torch import nn
from torch.nn import Module

class PositionalEncoding(Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DenseEmbedding(Module):

    def __init__(self, input_size, embedding_size,
                 hidden_size, dropout=0.1,
                 h_nonlinearity=nn.ReLU(),
                 dtype=torch.float32,
                 device=None):

        super().__init__()

        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)

        if isinstance(hidden_size, (int, float)):
            self.hidden_size = [int(hidden_size)]
        elif isinstance(hidden_size, (list, tuple)):
            self.hidden_size = list(hidden_size)

        self.embedding_size = embedding_size
        self.input_size = input_size

        if callable(h_nonlinearity):
            self.h_nonlinearity = len(self.hidden_size) * [h_nonlinearity]

        elif isinstance(nonlinearity, (list, tuple)):
            self.h_nonlinearity = h_nonlinearity

        if len(self.h_nonlinearity) != len(self.hidden_size):
            raise ValueError('h_nonlinearity needs to be the same size as `hidden_size`')

        if not isinstance(dropout, (list, tuple)):
            self.dropout = len(self.hidden_size) * [dropout]
        else:
            if len(dropout) != len(self.hidden_size):
                raise ValueError('`dropout` should be the same length '
                                 'as `hidden_size`.')

        hidden_layers = []

        in_features = input_size
        for hs, nl, p in zip(self.hidden_size, self.h_nonlinearity, self.dropout):
            
            hidden_layers.append(nn.Linear(in_features, hs))
            in_features = hs
            hidden_layers.append(nl)

            if p!= 0:
                hidden_layers.append(nn.Dropout(p))


        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(in_features=self.hidden_size[-1],
                                out_features=self.embedding_size)

    def forward(self, x):
        h = self.hidden_layers(x)
        output = self.output(h)
        return output


class PerformanceTransformerV1(Module):

    def __init__(self, input_size, output_size,
                 d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):

        super().__init__()

        self.input_embedding = DenseEmbedding(input_size=input_size,
                                              embedding_size=d_model,
                                              hidden_size=[1024])
        self.output_embedding = DenseEmbedding(input_size=output_size,
                                               embedding_size=d_model,
                                               hidden_size=[512])
        self.pe = PositionalEncoding(d_model=d_model)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)

        self.nhead = self.transformer.nhead
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size

        self.out = nn.Linear(d_model, output_size)


    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # embedd the input
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pe(src)

        if tgt is not None:
            tgt = self.output_embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pe(tgt)

            tgt = self.transformer(src, tgt,
                                   src_mask=src_mask,
                                   tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
        else:
            tgt = self.transformer.encoder(src, mask=src_mask,
                                           src_key_padding_mask=src_key_padding_mask)

            
        return self.out(tgt)


if __name__ == '__main__':

    
    input_size = 10
    embedding_size = 7
    hidden_size = [9, 8]
    output_size = 4
    src = torch.randn((1, 15, input_size), dtype=torch.float32)
    trg = torch.randn((1, 15, output_size), dtype=torch.float32)

    transformer = PerformanceTransformerV1(input_size, output_size)

    rec = transformer(src)
    

        
