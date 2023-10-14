
import datetime
import math
import numpy as np
import torch
from torch import nn
import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

class ETA_SGNN(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.SSL_model = opt.SSL_model 
        self.a1 = opt.a1
        self.a2 = opt.a2
        self.step = 2
        self.input_size = self.dim
        self.gate_size = 3 * self.dim
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.dim))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.dim))
        self.b_oah = Parameter(torch.Tensor(self.dim))
        
        self.w_ih1 = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh1 = Parameter(torch.Tensor(self.gate_size, self.dim))
        self.b_ih1 = Parameter(torch.Tensor(self.gate_size))
        self.b_hh1= Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_out = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_f = nn.Linear(self.dim, self.dim, bias=True)
        
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.dim, nhead=2,dim_feedforward=self.dim * 4)
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, 1)
        self.layer_norm1 = nn.LayerNorm(self.dim)
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        # self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Sequential(nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
                               nn.LeakyReLU(inplace=True),nn.Linear(self.dim, self.dim, bias=True))
                              
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask,item):
        mask = mask.float().unsqueeze(-1)
        len = hidden.shape[1]
        batch_size = hidden.shape[0]
        Item_emb = self.embedding(item)
        Item_emb = Item_emb.permute(1, 0, 2)
        S= self.layer_norm1(Item_emb)
        Item_emb = self.transformerEncoder(Item_emb)
        Item_emb = Item_emb.permute(1, 0, 2)
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        H = hidden + Item_emb
        hs = torch.sum(H * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)

        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)

        beta = beta * mask

        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)
        P = score(sess_emb_hgnn, sess_emb_lgcn)
        N = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        con_loss = torch.sum(-torch.log(1e-9 + torch.sigmoid(P)) - torch.log(1e-9 + (torch.cuda.FloatTensor(N.shape[0]).fill_(1) - torch.sigmoid(N))))
        return con_loss
    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        gi = F.linear(input_in, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy
    def GNNCell2(self, hidden):
        gi = F.linear(hidden, self.w_ih1, self.b_ih1)
        gh = F.linear(hidden, self.w_hh1, self.b_hh1)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy
    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)
        hidden_1 = h
        for i in range(1):
            hidden_1 = self.GNNCell(adj, hidden_1)

        h_local = self.local_agg(h, adj, mask_item)

        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        if self.SSL_model == 'graph':
            h_global2 = F.dropout(h_global, 0.4, training=self.training)
            h_global_1 = torch.sum(h_global,1)
            h_global_2 = torch.sum(h_global2,1)
            h_global_1 = self.linear_transform(h_global_1)
            h_global_2= self.linear_transform(h_global_2)
            loss_ssl = self.SSL(h_global_1, h_global_2)
        if self.SSL_model == 'GGNN':
            h_global_ = torch.sum(h_global,1)
            h_global_ = self.linear_transform(h_global_)        
            hidden_11 = torch.sum(hidden_1,1)
            hidden_11 = self.linear_transform(hidden_11)
            
            loss_ssl = self.SSL(h_global_, hidden_11)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + self.a2*h_global

        return output,loss_ssl


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    hidden, loss_ssl = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask,items),loss_ssl


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    for data in train_loader:
        model.optimizer.zero_grad()
        targets, scores, loss_ssl = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss_total = loss + 0.05*loss_ssl
        loss_total.backward()
        model.optimizer.step()
        total_loss += loss_total
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores,loss_ssl = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result

