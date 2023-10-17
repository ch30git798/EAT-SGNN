import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim
        self.block = 'sum'
       
        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        if self.block ==  'sum':
            self.w_3 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        if self.block ==  'cat':
            self.w_3 = nn.Parameter(torch.Tensor(2*self.dim, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            #alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1),neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        
        if self.block == 'cat':#w1 return to 
            # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
            #output = torch.cat([self_vectors, neighbor_vector], -1)
            output = torch.cat([self_vectors, neighbor_vector], -1)
            
            output = F.dropout(output, self.dropout, training=self.training)
            output = torch.matmul(output, self.w_3)
           
            output = output.view(batch_size, -1, self.dim)
           
            output = self.act(output)
        
            out2 = torch.mul(self_vectors,neighbor_vector)
            out2 = F.dropout(out2, self.dropout, training=self.training)
            out2 = torch.matmul(out2, self.w_4)
            out2 = out2.view(batch_size, -1, self.dim)
        
            out2 = self.act(output)
        
            output = output + out2
            
            return output
        if self.block == 'sum':
        
            output = torch.cat([self_vectors.unsqueeze(3), neighbor_vector.unsqueeze(3)], -1)
            output = torch.sum(output,-1)
            output2 = self_vectors * neighbor_vector

            
            output = F.dropout(output, self.dropout, training=self.training)
            output2 = F.dropout(output2, self.dropout, training=self.training)
            output = torch.matmul(output, self.w_3)
            output2 = torch.matmul(output2, self.w_4)
            O = output+output2
            
            O = O.view(batch_size, -1, self.dim)
            output = self.act(O)
      
            return output
