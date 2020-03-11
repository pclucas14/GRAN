import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn

from torch_scatter import scatter_add

SELF_LOOP = False

'''
Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''


class graph_decoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, decoder, mlp_in, output_theta, output_alpha):
        super(graph_decoder, self).__init__()
        self.decoder    = decoder
        self.mlp_in     = mlp_in
        self.output_theta = output_theta
        self.output_alpha = output_alpha
        self.self_loop  = SELF_LOOP

    def mix_bern_loss(self, log_theta, log_alpha, adj, lens):

        B, N, K = log_theta.size()
        adj = adj.squeeze(1)

        # we are going to need a mask to mask out nodes after the one being predicted
        # remember : we are predicting the connections for the node at index len[i] for i
        valid_edges = torch.zeros(B, N).to(adj.device)
        valid_edges[torch.arange(B), lens + 1] = 1
        valid_edges = 1 - valid_edges.cumsum(1)

        # build label
        label = adj[torch.arange(B), lens]

        ### individual loss for every mixture
        adj_loss = F.binary_cross_entropy_with_logits(\
                log_theta.view(-1, K), label.unsqueeze(-1).expand(-1, -1, K).view(-1, K), reduction='none')
        adj_loss = adj_loss.view(B, -1, K)

        # mask out padded noded / nodes after the one to be predicted
        adj_loss = adj_loss * valid_edges.unsqueeze(-1)

        # and reduce / sum per row
        adj_loss = adj_loss.sum(1)

        """ equivalent to
        out = torch.stack([F.binary_cross_entropy_with_logits(log_theta[:, :, i], label, reduction='none')
             for i in range(K)], -1)
        """

        ### build alphas

        # we want to pool together the mixtures coming from the same subgraph,
        # i.e. coming from the same row prediction
        log_alpha = log_alpha * valid_edges.unsqueeze(-1)

        # average over pooled nodes
        log_alpha = log_alpha.sum(1) / lens.view(-1, 1)
        log_alpha = F.log_softmax(log_alpha, -1)

        log_prob = -adj_loss + log_alpha
        log_prob = torch.logsumexp(log_prob, 1)
        return - log_prob.sum() / lens.sum().float()


    def sample(self, n_samples=64, max_node=100):

        with torch.no_grad():

            # build node features
            adj = torch.zeros(n_samples, max_node, max_node).to('cuda:0')

            # build ar mask
            attn_mask = torch.zeros(1, max_node, max_node).to(adj.device)
            valid_edges = torch.zeros(1, max_node).to(adj.device)

            # assuming no self loops, we can start at the second node (i = 1)
            ### TODO: put this bach
            for ii in range(1, 50): #max_node):

                # rows ii: are already zeros (line 82)
                node_feat = adj.float()

                # add dummy edges
                attn_mask[:, ii, :ii+1] = 1
                attn_mask[:, :ii+1, ii] = 1

                # add self loops
                # to not include self connection, uncomment this line
                attn_mask[:, torch.arange(ii), torch.arange(ii)] = 1

                usq = lambda x : x.unsqueeze(1)
                lens = torch.ones(n_samples).to(node_feat.device) * ii
                log_theta, log_alpha = self(usq(node_feat), usq(attn_mask), lens.long())

                # we are going to need a mask to mask out nodes after the one being predicted
                # remember : we are predicting the connections for the node at index len[i] for i
                # (include ii)
                valid_edges[:, :ii+1] = 1

                # mask out irrelevant tokens
                log_alpha = log_alpha * valid_edges.unsqueeze(-1)
                log_alpha = log_alpha.sum(1) / ii
                alpha     = F.softmax(log_alpha, -1)
                alpha     = torch.multinomial(alpha, 1).squeeze(1)

                log_theta = log_theta[torch.arange(n_samples), :, alpha]
                log_theta = log_theta[:, :ii+1]

                # Temperature tuning
                log_theta = log_theta * 2


                theta     = torch.sigmoid(log_theta)

                sampled_edges = torch.bernoulli(theta)

                adj[:, ii, :ii+1] = sampled_edges

                adj = (adj + adj.transpose(-2, -1)).clamp_(max=1)

            return adj


        # we can actually start our loop at i == 1 since we assume there is no self loops

    def forward(self, node_feat, attn_mask, lens):
        B, _, N, D = node_feat.size()

        # remove channel dim
        # TODO: move this in `preprocess`
        node_feat  = node_feat.squeeze(1)
        attn_mask    = attn_mask.squeeze(1)

        ### build node features (elementwise op)
        node_feat = self.mlp_in(node_feat)

        ### graph stuff
        x = self.decode(node_feat, attn_mask)

        predicted_node_feat  = x[torch.arange(B), lens]

        diff = predicted_node_feat.unsqueeze(1) - x

        """
        # botch (bs, N, D)
        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)
        """

        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)

        return log_theta, log_alpha


    def decode(self, x, attn_mask):
        # tgt = self.tgt_embed(tgt)
        return self.decoder(x, attn_mask)


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
        self.norm_x = LayerNorm(layer.size)

    def forward(self, x, attn_mask):
        for i, layer_x in enumerate(self.layers_x):
            # 1) we do the regular GAT-ish connection using the regular graph
            # this performs a x' = x + LayerNorm(self_att(x))
            x = layer_x(x, attn_mask)

        return self.norm_x(x)


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
        d_model=512, d_ff=2048, h=8, dropout=0.3): # 0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # MLP projection of Adj. matrix. Make sure you remove this when switching to VAE
    mlp_in = nn.Linear(100, d_model)

    output_theta = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_out))

    output_alpha = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_out))

    model = graph_decoder(
        Decoder(DecoderLayer(d_model, c(attn),
                             c(ff), dropout), N),
        mlp_in,
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

