import numpy as np
import torch

from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from basismixer.predictive_models import get_nonlinearity

class HierarchicalSequentialModel(Module):
    """
    Hierarchichal Model 
    """
    def __init__(self, seq_model):
        super().__init__()

        self.models = []
        if isinstance(seq_model, (list, tuple)):
            self.n_hierarchies = len(seq_model)
            for i, model in enumerate(seq_model):
                name = 'sm_{0}'.format(i)
                setattr(self, name, model)

                self.models.append(model)
                
        elif isinstance(seq_model, Module):
            self.sm_0 = seq_model
            self.n_hierarchies = 1
            self.models.append(model)

        else:
            raise ValueError(('`seq_model` must be an instance or a list of instances of '
                              '`torch.nn.Module`, but given {0}').format(type(seq_model)))

    def forward(self, input, hierarchy_idxs, *args, **kwargs):
        assert len(hierarchy_idxs) == self.n_hierarchies - 1

        
        out = self.models[0](input, *args, **kwargs)
        if isinstance(out, (list, tuple)):
            out = out[0]
        outputs = [out]
        for hi, model in enumerate(self.models[1:]):
            out = model(out[hierarchy_idxs[hi], :, :], *args, **kwargs)
            if isinstance(out, (list, tuple)):
                out = out[0]
            outputs.append(out)
        return out, outputs

class ContextAttention(Module):
    """
    Context Attention Model from Virtuoso Net
    """
    def __init__(self, size, n_heads=6, nl=nn.Tanh()):

        super().__init__()
        self.n_heads = n_heads
        self.linear = nn.Linear(in_features=size,
                                out_features=size)
        self.nl = get_nonlinearity(nl)
        if size % n_heads != 0:
            raise ValueError("size must be dividable by n_heads", size, n_heads)
        self.head_size = int(size // n_heads)
        self.context_vector = torch.nn.Parameter(torch.Tensor(self.n_heads, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

        # shortcut
        self.u_c = self.context_vector

    def forward(self, input):
        bz, sl, sz = input.shape
        u_t = self.nl(self.linear(input))
        if self.head_size > 1:
            u_t_split = u_t.split(self.head_size, dim=-1)
            ht_split = input.split(self.head_size, dim=-1)
            a_i = torch.cat([torch.matmul(s, uc)
                             for s, uc in zip(u_t_split, self.u_c)], dim=0).reshape(self.n_heads, -1)
            a_i = F.softmax(a_i, -1)
            m_i = [sum([a*h for a, h in zip(ai, hts)]) for ai, hts in zip(a_i, ht_split)]
            m = torch.cat(m_i, dim=-1).reshape(1, sl, -1)
        else:
            # TODO: This is not correct... Fix?
            a_i = F.softmax(u_t, 1)
            m_i = a_i * input
            m = torch.sum(m_i, dim=1)
        return m

class HierarchicalAttentionModel(Module):

    def __init__(self, seq_model, n_heads):
        super().__init__()

        self.sm = seq_model
        self.attention = ContextAttention(size=self.sm.output_size,
                                          n_heads=n_heads)


    def forward(self, inputs, attention_idxs):
        
        attention_output = []
        for aix in attention_indices:
            at_input = torch.cat([o[ai, :, :] for o, ai in zip(inputs, aix)], dim=0)
            attention_output.append(self.attention(at_input))

        attention_output = torch.cat(attention_output, dim=0)

        output, _ = self.sm(attention_output)

        return output
