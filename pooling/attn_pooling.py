import torch
from torch.nn import Linear

def graph_attn_op_batched(q, k, v, batch, batch_size):
    '''
    Returns a padded tensor of shape (batch_size, max_node_number, embedding_dim)
    '''
    attn_maps = []
    values = []
    for i in range(batch_size):
        attn_map = q[batch == i].matmul(k[batch == i].T)/q.shape[0]
        attn_maps.append(attn_map)
        values.append(v[batch == i])

    padded_attn_maps = torch.nested.as_nested_tensor(attn_maps).to_padded_tensor(0.)
    padded_values = torch.nested.as_nested_tensor(values).to_padded_tensor(0.)

    padded_values.shape
    return torch.matmul(padded_attn_maps, padded_values)

class GraphSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, inner_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.q = torch.nn.Linear(input_dim, inner_dim)
        self.k = torch.nn.Linear(input_dim, inner_dim)
        self.v = torch.nn.Linear(input_dim, inner_dim)
        self.out_proj = torch.nn.Linear(inner_dim, input_dim, bias=False)
    
    def forward(self, x, batch):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        out = graph_attn_op_batched(q, k, v, batch, batch.max() + 1)
        return self.out_proj(out)

class GraphMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, inner_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.heads = torch.nn.ModuleList([
            GraphSelfAttention(input_dim, inner_dim) for _ in range(num_heads)
        ])
        self.out_proj = torch.nn.Linear(inner_dim, input_dim, bias=False)

    def forward(self, x, batch):
        head_outputs = [head.forward(x, batch) for head in self.heads]
        # Concatenate along the last dimension
        out = torch.cat(head_outputs, dim=-1)
        return self.out_proj(out)


def global_attn_pool_sample(x, batch, attn_layer):
    x = attn_layer(x, batch)

    return x[:, 0, :]

def global_attn_pool_mean(x, batch, attn_layer, guess_tol=1e-5):
    '''
    Tries to figure out the true shape by guessing the padding
    '''
    x = attn_layer(x, batch)
    is_not_zero = x.abs() > guess_tol
    is_not_padding = is_not_zero.sum(dim=-1) > 0
    num_not_padding = is_not_padding.sum(dim=-1)

    return x.sum(dim=-2) / num_not_padding.unsqueeze(-1)