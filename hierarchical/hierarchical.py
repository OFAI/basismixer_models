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
            # at_input = []
            # for o, ai in zip(inputs, aix):
            #     at_input.append(o[ai, :, :])

            # at_input = torch.cat(at_input, dim=0)
            attention_output.append(self.attention(at_input))

        attention_output = torch.cat(attention_output, dim=0)

        output, _ = self.sm(attention_output)

        return output
        


    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from transformer.transformer import PerformanceTransformerV1
    seq_len = 1000
    input_size = 200
    x = torch.randn((seq_len, 1, input_size), requires_grad=True)
    attention = ContextAttention(12, n_heads=2)
    
    n_partitions = np.random.randint(5, 10)

    partitions = np.random.choice(np.arange(1, seq_len), n_partitions, replace=False)
    partitions.sort()

    hierarchy_idxs = np.zeros((seq_len, 3))

    hierarchy_idxs[-1] = 1
    hierarchy_idxs[:, 0] = 1
    hierarchy_idxs[partitions - 1, 1] = 1

    n_partitions = np.random.randint(2, max(n_partitions, 3))
    partitions = np.random.choice(partitions, n_partitions, replace=False)
    partitions.sort()
    hierarchy_idxs[partitions -1, 2] = 1

    sm_0 = nn.LSTM(input_size, hidden_size=6, bidirectional=True)
    # sm_1 = nn.LSTM(input_size=int(6 * (1 + sm_0.bidirectional)), hidden_size=6, num_layers=1, bidirectional=True)
    # sm_2 = nn.LSTM(input_size=int(6 * (1 + sm_1.bidirectional)), hidden_size=6, num_layers=1, bidirectional=True)
    # sm_3 = nn.LSTM(input_size=int(6 * (1 + sm_2.bidirectional)), hidden_size=6, num_layers=1, bidirectional=True)
    # sm_0 = PerformanceTransformerV1(input_size=input_size, output_size=12)
    sm_1 = PerformanceTransformerV1(input_size=12, output_size=12)
    sm_2 = PerformanceTransformerV1(input_size=12, output_size=12)
    sm_3 = PerformanceTransformerV1(input_size=12, output_size=12)
    
    

    time_steps = np.unique(hierarchy_idxs[:, 1])
    # time_steps.sort()

    unique_idxs = [np.where(hierarchy_idxs[:, 1] == u)[0] for u in time_steps]
    h_idxs = np.zeros(len(x), dtype=np.int)

    h_prev = None
    hierarchical_idxs = []
    try:
        o, _ = sm_0(x, h_prev)
    except:
        o = sm_0(x)
    
    h_idxs = np.where(hierarchy_idxs[:, 0] == 1)[0]
    hierarchical_idxs.append(h_idxs)
    try:
        o, _ = sm_1(o[h_idxs, :, :])
    except:
        o = sm_1(o[h_idxs, :, :])
    h_idxs = np.where(hierarchy_idxs[h_idxs, 1] == 1)[0]
    hierarchical_idxs.append(h_idxs)
    try:
        o, _ = sm_2(o[h_idxs, :, :])
    except:
        o = sm_2(o[h_idxs, :, :])
    h_idxs = np.where(hierarchy_idxs[h_idxs, 2] == 1)[0]
    hierarchical_idxs.append(h_idxs)

    try:
        o, _ = sm_3(o[h_idxs, :, :])
    except:
        o = sm_3(o[h_idxs, :, :])

    hsm = HierarchicalSequentialModel([sm_0, sm_1, sm_2, sm_3])
    out, outputs = hsm(x, hierarchical_idxs)
    at = attention(out)

    n_divisions = 3
    attention_idxs = np.random.choice(np.arange(n_divisions), len(x), replace=True)
    attention_idxs.sort()

    u_ais = np.unique(attention_idxs)

    attention_idxs_0 = [np.where(attention_idxs == u)[0] for u in u_ais]
    attention_idxs_1 = attention_idxs_0
    attention_indices = [ix for ix in zip(attention_idxs_0, attention_idxs_1)]

    

    attention_output = []
    for aix in attention_indices:
        at_input = []
        for o, ai in zip(outputs, aix):
            at_input.append(o[ai, :, :])

        at_input = torch.cat(at_input, dim=0)
        attention_output.append(attention(at_input))

    attention_output = torch.cat(attention_output)

    att_sm_0 = nn.LSTM(12, hidden_size=6, num_layers=1, bidirectional=True)

    
    # for o, aix in zip(outputs, attention_indices):
    #     at_input.append(o[aix, :, :])
    # at_input = torch.cat([o[aix, :, :] for o, aix in zip(outputs, attention_indices)], 0)
    
    
    


    

    
    
    
    # for s, uc in zip(ht_split, u_c):
    #     ai = torch.matmul(s, uc)
    #     a_i.append(ai)
    
