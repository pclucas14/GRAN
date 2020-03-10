import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn

from torch_scatter import scatter_add

SELF_LOOP = True
CHEAT = True
CHEAT_DIAG = True

'''
Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''


class graph_decoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, decoder, mlp_in, edge_mlp, output_theta, output_alpha):
        super(graph_decoder, self).__init__()
        self.decoder    = decoder
        self.mlp_in     = mlp_in
        self.edge_mlp   = edge_mlp
        self.output_theta = output_theta
        self.output_alpha = output_alpha
        self.self_loop  = SELF_LOOP

    def mix_bern_loss(self, log_theta, log_alpha, label, edge_idx):
        log_theta = log_theta
        log_alpha = log_alpha

        K = log_theta.size(-1)
        edge_idx_exp = edge_idx.unsqueeze(1).expand(-1, K)

        # individual loss for every mixture
        adj_loss = F.binary_cross_entropy_with_logits(\
                log_theta, label.view(-1, 1).expand(-1, K), reduction='none')
        reduce_adj_loss = scatter_add(adj_loss, edge_idx, 0)

        reduce_log_alpha = scatter_add(log_alpha, edge_idx, 0)
        count = scatter_add(torch.ones_like(edge_idx), edge_idx, dim=0)

        """ NOTE: edge_idx does leaps (e.g
        (Pdb) edge_idx[1020:1060]
        tensor([ 45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,
                 45, 101, 102, 102, 103, 103, 103, 104, 104, 104, 104, 105, 105, 105,
                105, 105, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107],
               device='cuda:0')
        this is because the indices are assigned before filtering out the padded rows
        we remove them
        """

        valid_idx = count.nonzero().squeeze()

        reduce_log_alpha = reduce_log_alpha[valid_idx]
        reduce_adj_loss  = reduce_adj_loss[valid_idx]
        count            = count[valid_idx].unsqueeze(-1)

        # depending on the row, different amount of nodes are added up to build alphas and thetas
        # it's important to that the average of the alphas to make sure were always on the same scale
        log_alpha  = reduce_log_alpha / count

        ### HERE
        log_alpha = F.log_softmax(log_alpha, -1)

        # return reduce_adj_loss.mean(1).sum() / log_theta.shape[0]
        # log_alpha = torch.Tensor([1./20.]).to(log_alpha.device).view(1, 1).expand(-1, 20)
        # log_alpha = log_alpha.log()

        log_prob = -reduce_adj_loss + log_alpha
        log_prob = torch.logsumexp(log_prob, 1)
        return - log_prob.sum() / log_theta.shape[0]


    def sample(self, n_samples=64, max_node=100):

        with torch.no_grad():

            # build node features
            node_feat = torch.zeros(n_samples, max_node, max_node).to('cuda:0')

            # build ar mask
            ar_mask = torch.tril(torch.ones(max_node, max_node)).to(node_feat.device)
            diag_idx = torch.arange(max_node)
            ar_mask[diag_idx, diag_idx] = 0
            ar_mask = ar_mask.unsqueeze(0).expand(n_samples, -1, -1)

            # build lt_adj_mat
            lt_adj_mat = node_feat.clone().long()

            if SELF_LOOP:
                # SELF EDGES
                ar_mask[:, torch.arange(100), torch.arange(100)] = 1
                lt_adj_mat[:, torch.arange(100), torch.arange(100)] = 1

            for ii in range(1, max_node):

                usq = lambda x : x.unsqueeze(1)
                log_theta, log_alpha = self(usq(node_feat), usq(ar_mask), usq(lt_adj_mat))

                ### sample_alpha values
                log_alpha = log_alpha[:, ii, :ii]

                # the number of dummy edges is simply (ii+1) - 1, since we don't have self loops
                log_alpha = log_alpha / ii
                ### HERE normalize after div.
                log_alpha = F.log_softmax(log_alpha, -1)

                # alpha : (n_samples * ii, )
                alpha = torch.multinomial(log_alpha.view(-1, log_alpha.size(-1)).exp(),
                        num_samples=1).squeeze(-1)#.reshape(log_alpha.shape[:2])

                print('unique alphas ', alpha.unique())

                ### build theta values
                log_theta = log_theta[:, ii, :ii]
                log_theta = log_theta.reshape(-1, log_theta.size(-1))[torch.arange(alpha.size(0)), alpha]
                theta = F.sigmoid(log_theta).reshape(log_alpha.shape[:2])

                ### only do a single mixture
                # theta = F.sigmoid(log_theta[:, ii, :ii, 0])

                sampled_edges = torch.bernoulli(theta)

                ### update the current graph data structure

                # attention mask
                lt_adj_mat[:, ii, :ii] = sampled_edges.long()

                if SELF_LOOP:
                    lt_adj_mat[:, ii, ii] = 1
                else:
                    node_feat = lt_adj_mat.clone().float()

                if CHEAT:
                    node_feat = lt_adj_mat.clone().float()

            adj_mat = lt_adj_mat + lt_adj_mat.transpose(-2,-1)

            return adj_mat


        # we can actually start our loop at i == 1 since we assume there is no self loops

    def forward(self, node_feat, ar_mask, lt_adj_mat):

        # TODO: investigate this:
        # actually, add self loops but zero features

        if SELF_LOOP or CHEAT_DIAG:
            # SELF EDGES
            ### HERE
            if not CHEAT: node_feat = node_feat * 0
            ar_mask[:, :, torch.arange(100), torch.arange(100)] = 1
            lt_adj_mat[:, :, torch.arange(100), torch.arange(100)] = 1

            ar_mask = lt_adj_mat = torch.ones_like(ar_mask)

        """
        node_feat : bs, 1, N, N adjacency matrix

        given the adjacency matrix, ADJ=
            [ 0 1 0 0 1 ]
            [ 1 0 1 0 0 ]
            [ 0 1 1 1 0 ]
            [ 0 0 1 0 0 ]
            [ 1 0 0 0 0 ]

        we build
        1) the node features, which is the lower triangular part of Adj
            [ 0 0 0 0 0 ]
            [ 1 0 0 0 0 ]
            [ 0 1 1 0 0 ]
            [ 0 0 1 0 0 ]
            [ 1 0 0 0 0 ]

        2) the attention masks. We need to be careful of avoiding self loops,
           since the node features for node i is the target of node i
           a) the autoregressive mask for the dummy edges, ARM=
            [ 0 0 0 0 0 ]
            [ 1 0 0 0 0 ]
            [ 1 1 0 0 0 ]
            [ 1 1 1 0 0 ]
            [ 1 1 1 1 0 ]

           b) the autoregressive attention mask
           which is equal to ARM * ADJ
            [ 0 0 0 0 0 ]
            [ 1 0 0 0 0 ]
            [ 0 1 0 0 0 ]
            [ 0 0 1 0 0 ]
            [ 1 0 0 0 0 ]
        """

        # remove channel dim
        # TODO: move this in `preprocess`
        node_feat  = node_feat.squeeze(1)
        ar_mask    = ar_mask.squeeze(1)
        lt_adj_mat = lt_adj_mat.squeeze(1)

        ### build node features (elementwise op)
        # TAG:
        node_feat_ = self.mlp_in(node_feat)

        ### graph stuff
        x, y = self.decode(node_feat, ar_mask, lt_adj_mat)

        ### now that we have node features for every node, we need to build the edge difference
        edge_x, edge_y = self.edge_mlp(torch.cat((x, y))).chunk(2)

        # (bs, N, N, D)
        all_diffs = edge_y.unsqueeze(1) - edge_x.unsqueeze(2)

        # we only need the lower triangular part of the (_, N, N, _) matrix
        # TODO: don't push useless calculations

        return self.output_theta(all_diffs), self.output_alpha(all_diffs)


    def decode(self, x, ar_mask, lt_adj_mat):
        # tgt = self.tgt_embed(tgt)
        return self.decoder(x, ar_mask, lt_adj_mat)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        #### HERE
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        xx = 1
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers_x = clones(layer, N)
        self.layers_y = clones(layer, N)
        self.norm_x = LayerNorm(layer.size)
        self.norm_y = LayerNorm(layer.size)

    def forward(self, x, ar_mask, lt_adj_mat):
        return x, torch.ones_like(x) * -1
        import pdb; pdb.set_trace()
        for i, (layer_x, layer_y) in enumerate(zip(self.layers_x, self.layers_y)):
            # 1) we do the regular GAT-ish connection using the regular graph
            # this performs a x' = x + LayerNorm(self_att(x))
            x = layer_x(x, lt_adj_mat)

            if i == 0:
                y = x # layer_y(x, ar_mask)
            else:
                #### HERE : remove the left part
                y = y# + layer_y(x, ar_mask)
                #y = layer_y(x, ar_mask)

        return self.norm_x(x), self.norm_y(y)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        "Follow Figure 1 (right) for connections."

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

306532
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def make_model(d_out=100, N=6,
        d_model=512, d_ff=2048, h=8, dropout=0.5): # 0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # MLP projection of Adj. matrix. Make sure you remove this when switching to VAE
    mlp_in = nn.Linear(100, d_model)

    edge_mlp = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU())

    output_theta = nn.Linear(128, 20)
    output_alpha = nn.Linear(128, 20)

    model = graph_decoder(
        Decoder(DecoderLayer(d_model, c(attn),
                             c(ff), dropout), N),
        mlp_in,
        edge_mlp,
        output_theta,
        output_alpha)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask * subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def greedy_decode(model, max_len, start_symbol=2):
    raise NotImplementedError

if __name__ == '__main__':

    d_out = 50
    d_model = 128
    d_ff = 512
    h = 8

    model = make_model(d_out=d_out, d_model=d_model, d_ff=d_ff, h=h).cuda()
    input = torch.FloatTensor(16, 51, d_model).cuda()
    out   = model(input) #, mask, mask)

    print('number of parameters : {}'.format(sum([np.prod(x.shape) for x in model.parameters()])))
    # import pdb; pdb.set_trace()
    xx = 1

