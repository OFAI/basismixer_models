import numpy as np
import torch

from torch import nn
from torch.nn import Module
import torch.nn.functional as F

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

    def forward(self, input, hierarchy_idxs):
        assert len(hierarchy_idxs) == self.n_hierarchies - 1
        out, _ = self.models[0](input)
        outputs = [out]
        for hi, model in enumerate(self.models[1:]):
            out, _ = model(out[hierarchy_idxs[hi], :, :])
            outputs.append(out)

        return out, outputs

class ContextAttention(Module):
    """
    Context Attention Model from Virtuoso Net

    Adapted from https://github.com/jdasam/virtuosoNet/blob/master/nnModel.py
    """
    def __init__(self, size, n_heads=6, nl=nn.Tanh()):

        super().__init__()
        self.n_heads = n_heads
        self.linear = nn.Linear(in_features=size,
                                out_features=size)
        if nl is None:
            self.nl = lambda x: x
        elif callable(nl):
            self.nl = nl
        else:
            raise ValueError('`nl` must be a callable or None')
        
        if size % n_heads != 0:
            raise ValueError("size must be dividable by n_heads", size, n_heads)
        self.head_size = int(size // n_heads)
        self.context_vector = torch.nn.Parameter(torch.Tensor(self.n_heads, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

        # shortcult
        self.u_c = self.context_vector

    def forward(self, input):
        bz, sl, sz = input.shape
        u_t = self.nl(self.linear(input))
        # import pdb
        # pdb.set_trace()
        if self.head_size > 1:
            u_t_split = u_t.split(self.head_size, dim=-1)
            ht_split = input.split(self.head_size, dim=-1)

            
            a_i = torch.cat([torch.matmul(s, uc)
                             for s, uc in zip(u_t_split, self.u_c)], dim=0).reshape(self.n_heads, -1)
            # a_i = []
            # for s, uc in zip(u_t_split, self.u_c):
            #     ai = torch.matmul(s, uc)
            #     a_i.append(ai)
            
            # a_i = torch.cat(a_i, dim=0).reshape(self.n_heads, -1)
            # import pdb
            # pdb.set_trace()
            a_i = F.softmax(a_i, -1)
            m_i = [sum([a*h for a, h in zip(ai, hts)]) for ai, hts in zip(a_i, ht_split)]
            m = torch.cat(m_i, dim=-1).reshape(1, sl, -1)
        else:
            a_i = F.softmax(u_t, 1)
            m_i = a_i * input
            m = torch.sum(m_i, dim=1)
        return m

def dev():

    seq_len = 100
    input_size = 12
    x = torch.randn((seq_len, 1, input_size), requires_grad=True)
    attention = ContextAttention(input_size)

    u = attention.linear(x)
    if attention.head_size > 1:
        u_split = torch.cat(u.split(split_size=attention.head_size, dim=-1), dim=0)
    # att = attention(x)

    if False:
        n_partitions = np.random.randint(5, 10)

        partitions = np.random.choice(np.arange(1, seq_len), n_partitions, replace=False)
        partitions.sort()

        hierarchy_idxs = np.zeros((seq_len, 3))

        hierarchy_idxs[-1] = 1
        hierarchy_idxs[:, 0] = 1
        hierarchy_idxs[partitions - 1, 1] = 1

        n_partitions = np.random.randint(2, max(n_partitions, 3))
        partitions = np.random.choice(partitions, n_partitions, replace=False)

        hierarchy_idxs[partitions -1, 2] = 1

        sm_0 = nn.LSTM(input_size, hidden_size=3, bidirectional=True)
        sm_1 = nn.LSTM(input_size=int(3 * (1 + sm_0.bidirectional)), hidden_size=3, num_layers=1, bidirectional=True)
        sm_2 = nn.LSTM(input_size=int(3 * (1 + sm_1.bidirectional)), hidden_size=3, num_layers=1, bidirectional=True)
        sm_3 = nn.LSTM(input_size=int(3 * (1 + sm_2.bidirectional)), hidden_size=3, num_layers=1, bidirectional=True)

        time_steps = np.unique(hierarchy_idxs[:, 1])
        # time_steps.sort()

        unique_idxs = [np.where(hierarchy_idxs[:, 1] == u)[0] for u in time_steps]
        h_idxs = np.zeros(len(x), dtype=np.int)

        h_prev = None
        hierarchical_idxs = []
        o, _ = sm_0(x, h_prev)
        h_idxs = np.where(hierarchy_idxs[:, 0] == 1)[0]
        hierarchical_idxs.append(h_idxs)
        o, _ = sm_1(o[h_idxs, :, :])
        h_idxs = np.where(hierarchy_idxs[h_idxs, 1] == 1)[0]
        hierarchical_idxs.append(h_idxs)
        o, _ = sm_2(o[h_idxs, :, :])
        h_idxs = np.where(hierarchy_idxs[h_idxs, 2] == 1)[0]
        hierarchical_idxs.append(h_idxs)
        o, _ = sm_3(o[h_idxs, :, :])

        hsm = HierarchicalSequentialModel([sm_0, sm_1, sm_2, sm_3])
        out = hsm(x, hierarchical_idxs)

def dev_attention():
    head_size = 3
    n_heads = 2
    ht = torch.tensor([[[1,2,3,4, 5, 6]], [[4, 5, 7, 9, 4, 7]], [[10, 9, 4, 7, 9, 1]]], dtype=torch.float32)

    sl, bz, size = ht.shape

    u_c = torch.tensor([[[4], [7], [9]], [[1], [2], [7]]], dtype=torch.float32)

    ht_split = ht.split(head_size, dim=-1)

    a_i = torch.cat([torch.matmul(s, uc) for s, uc in zip(ht_split, u_c)], 0).reshape(-1, head_size)

    a_i = F.softmax(a_i, -1)

    m = torch.cat([sum([a*h for a, h in zip(ai, hts)]) for ai, hts in zip(a_i, ht_split)], -1).reshape(1, bz, -1)

    attention = ContextAttention(6, n_heads, nl=None)
    attention.context_vector.data = u_c
    attention.linear.weight.data = torch.eye(6)
    attention.linear.bias.data *= torch.zeros_like(attention.linear.bias.data)
    ma = attention(ht)

    assert (np.allclose(ma.data.detach().numpy(), m.data.detach().numpy()))

    
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    seq_len = 100
    input_size = 12
    x = torch.randn((seq_len, 1, input_size), requires_grad=True)
    attention = ContextAttention(6, n_heads=2)

    n_partitions = np.random.randint(5, 10)

    partitions = np.random.choice(np.arange(1, seq_len), n_partitions, replace=False)
    partitions.sort()

    hierarchy_idxs = np.zeros((seq_len, 3))

    hierarchy_idxs[-1] = 1
    hierarchy_idxs[:, 0] = 1
    hierarchy_idxs[partitions - 1, 1] = 1

    n_partitions = np.random.randint(2, max(n_partitions, 3))
    partitions = np.random.choice(partitions, n_partitions, replace=False)

    hierarchy_idxs[partitions -1, 2] = 1

    sm_0 = nn.LSTM(input_size, hidden_size=6, bidirectional=False)
    sm_1 = nn.LSTM(input_size=int(6 * (1 + sm_0.bidirectional)), hidden_size=6, num_layers=1, bidirectional=False)
    sm_2 = nn.LSTM(input_size=int(6 * (1 + sm_1.bidirectional)), hidden_size=6, num_layers=1, bidirectional=False)
    sm_3 = nn.LSTM(input_size=int(6 * (1 + sm_2.bidirectional)), hidden_size=6, num_layers=1, bidirectional=False)

    time_steps = np.unique(hierarchy_idxs[:, 1])
    # time_steps.sort()

    unique_idxs = [np.where(hierarchy_idxs[:, 1] == u)[0] for u in time_steps]
    h_idxs = np.zeros(len(x), dtype=np.int)

    h_prev = None
    hierarchical_idxs = []
    o, _ = sm_0(x, h_prev)
    h_idxs = np.where(hierarchy_idxs[:, 0] == 1)[0]
    hierarchical_idxs.append(h_idxs)
    o, _ = sm_1(o[h_idxs, :, :])
    h_idxs = np.where(hierarchy_idxs[h_idxs, 1] == 1)[0]
    hierarchical_idxs.append(h_idxs)
    o, _ = sm_2(o[h_idxs, :, :])
    h_idxs = np.where(hierarchy_idxs[h_idxs, 2] == 1)[0]
    hierarchical_idxs.append(h_idxs)
    o, _ = sm_3(o[h_idxs, :, :])

    hsm = HierarchicalSequentialModel([sm_0, sm_1, sm_2, sm_3])
    out, outputs = hsm(x, hierarchical_idxs)
    at = attention(out)
    
    
    # for s, uc in zip(ht_split, u_c):
    #     ai = torch.matmul(s, uc)
    #     a_i.append(ai)
    
