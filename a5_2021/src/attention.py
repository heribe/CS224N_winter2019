import math
import logging
import re

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        print("use causal")
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

"""
Write your SynthesizerAttention below.
Hint: paste over the CausalSelfAttention above and modify it minimally.
"""

class SynthesizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("use synthe")
        assert config.n_embd % config.n_head == 0
        # NEW learnable weights
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))  #(hs,BS)
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        # TODO [part g]: Write your SynthesizerAttention below.
        #   Do not modify __init__().
        # Hints:
        #   - Paste over the CausalSelfAttention above and modify it minimally.
        #   - Consider especially the parameters self.w1, self.w2 and self.b2.
        #       How do these map to the matrices in the handout?
        B,nBS,ES = x.size()
        # BS = self.block_size-1
        # print("B,nBS,ES,BS",x.shape,BS)
        # Yi = softmax(ReLU(XAi + b1)Bi + b2)(XVi),
        # t1 = RelU(XAi+b1)
        t1 = F.relu(self.w1(x)).view(B,nBS,self.n_head,ES//self.n_head).transpose(1,2)# (B,nBS,hn,hs)->(B,hn,nBS,hs)
        # t2 = t1@Bi+b2
        w2 = self.w2.view(1,1,self.w2.shape[0],self.w2.shape[1])[...,:nBS]
        b2 = self.b2.view(1,1,1,self.b2.shape[0])[...,:nBS]
        t2 = t1@w2+b2#(B,hn,nBS,hs)*(1,1,hs,BS) -> (B,hn,nBS,BS)
        # att = softmax(t2)
        # print("t2.size: ",t2.shape)
        att = t2.masked_fill(self.mask[:,:,:nBS,:nBS] == 0, -1e10)
        att = F.softmax(att,dim=-1) # (B,hn,nBS,BS)
        att = self.attn_drop(att)
        # y = att@XVi
        v = self.value(x).view(B,nBS,self.n_head,ES//self.n_head).transpose(1,2) #(B,nBS,hn,hs)->#(B,hn,nBS,hs)
        # print('v.size: ',v.shape)
        # print('att.size: ',att.shape)
        # att = att.transpose(-2,-1) #(B,hn,BS,nBS)
        y = att@v  # (B,hn,BS,nBS)*(B,hn,nBS,hs)->(B,hn,BS,hs)
        # y->Y
        y = y.transpose(1,2).reshape(B,nBS,ES)
        y = self.resid_drop(self.proj(y))
        return y
        # raise NotImplementedError
