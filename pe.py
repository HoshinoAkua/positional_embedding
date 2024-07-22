import torch
import torch.nn as nn
'''
key(j) 到 query(i) 的相对距离 计算公式为: 
b(i,j) = log(c(i-j)+1)/log(c*max(L,i)+1)
我实际用的是 b(i,j) = i-j/i
'''
import torch
import torch.nn as nn
'''
key(j) 到 query(i) 的相对距离 计算公式为: 
b(i,j) = log(c(i-j)+1)/log(c*max(L,i)+1)
我实际用的是 b(i,j) = i-j/i
'''
class FIRE(nn.Module):
    def __init__(self, num_heads = 12, mlp_width = 32, init_c = 0.1, init_L = 512., eps = 1e-6):
        super(FIRE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1,mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads)
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L),requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps
        
    def forward(self, query_length, key_length, device):
        #x.size = [batch_size, num_heads, seq_length, hidden_dim=768/12]
        
        key_positions = torch.arange(key_length, dtype= torch.float,
                                 device=device)
        query_positions = torch.arange(query_length, dtype=torch.float,
                                       device=device)
        rel_distance = query_positions[:,None] - key_positions[None,:]
        
        '''
        None是为了升维度, 然后可以利用广播计算.
        position = torch.arange(3,dtype= torch.float)
        rel = position[:,None]-position[None:]
        rel: tensor([[ 0., -1., -2.],
                    [ 1.,  0., -1.],
                    [ 2.,  1.,  0.]])
        '''
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(key_positions, threshold)
        rel_distance = torch.log(
            torch.abs(self.c * rel_distance) + 1
            )
        pos_normalizer = torch.log(
            torch.abs(self.c * pos_normalizer) + 1
            ) + self.eps
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        
        # normalized_distance = (rel_distance)/(positions+self.eps)[:,None]
        # fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        
        fire_bias = fire_bias.unsqueeze(0).permute(0,3,1,2)
        return fire_bias
    

class CoPE(nn.Module):
    def __init__(self, npos_max, hidden_dim, n_heads):
        super().__init__()
        self.npos_max = npos_max
        self.n_head = n_heads
        self.attn_head_size = int(hidden_dim/n_heads)
        self.pos_emb = nn.Parameter(torch.ones(1,npos_max, hidden_dim))
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, query, attn_logits): #attn_logits的输入是[batch, n_head, query_length, key_length]
        gates = torch.sigmoid(attn_logits)

        #gates = torch.ones(size = attn_logits.shape, device= attn_logits.device)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1) #先反转, 再累加, 再反转
        #pos = gates.cumsum(dim=-1)
        pos = pos.clamp(max=self.npos_max-1) #pos矩阵已经是每一个query对keys的距离, 然后我们限制这个距离不能超过npos_max
        
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        pos_emb = self._split_heads(self.pos_emb, self.n_head, self.attn_head_size)
        logits_int = torch.matmul(query , pos_emb.transpose(-1,-2))
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil*w + logits_floor*(1-w)
class RawCoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.randn(1, head_dim,npos_max))
    def forward(self, query, attn_logits): #attn_logits的输入是[batch, n_head, query_length, key_length]
        gates = torch.sigmoid ( attn_logits )
        #gates = torch.ones(size = attn_logits.shape)
        #pos = gates.flip(-1).cumsum(dim=-1).flip(-1) #先反转, 再累加, 再反转
        pos = gates.cumsum(dim=-1)
        pos = pos.clamp(max=self.npos_max-1) #pos矩阵已经是每一个query对keys的距离, 然后我们限制这个距离不能超过npos_max
        pos_ceil = pos.ceil().long ()
        pos_floor = pos.floor().long ()
        logits_int = torch . matmul ( query , self . pos_emb )
        logits_ceil = logits_int.gather(-1,pos_ceil )
        logits_floor = logits_int.gather(-1,pos_floor )
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)
class CoPEWithFIRE(nn.Module):
    def __init__(self, n_head=12 ,mlp_width = 32, init_c = 0.1, init_L = 512., eps = 1e-6):
        super().__init__()
        self.n_head = n_head
        self.input = nn.Linear(1,mlp_width)
        self.mlp_list = nn.ModuleList(
            nn.Sequential(
            self.input,
            nn.ReLU(),
            nn.Linear(mlp_width, 1)
        ) for _ in range(n_head)
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L),requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, attn_logits): #attn_logits的输入是[batch, n_head, query_length, key_length]
        gates = torch.sigmoid(attn_logits)
        
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1) #先反转, 再累加, 再反转
        
        numerator = torch.log(
            torch.abs(self.c * pos)+1
        )
        max_pos = pos[:,:,:,0].unsqueeze(-1)
        threshold = torch.abs(self.L_multiplier*self.init_L)
        denominator = torch.log(
            torch.abs(self.c * max_pos.clamp(max=threshold))+1
        )+self.eps 
        
        normalized_distance = (numerator/denominator).split(1, dim=1) #拆分完形状为 [batch_size, num_head=1, query_length, key_length]
        distance_list = [distance.squeeze(1).unsqueeze(-1) for distance in normalized_distance]# [batch, query, key, 1]
        bias_list = []
        for mlp, distance in zip(self.mlp_list,distance_list):
            bias = mlp(distance).permute(0,3,1,2) # [batch,1, query, key]
            bias_list.append(bias)
        fire_bias = torch.cat(bias_list, dim=1)
        return fire_bias


