#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch

# Make sure that your module uses two nn.Linear layers (this is important for the autograder)
class Highway(nn.Module):
    #def __init__(self,embed_word,prob):
    def __init__(self,embed_word):
        super(Highway,self).__init__()
        self.linear1 = nn.Linear(embed_word,embed_word,bias=True)
        self.linear2 = nn.Linear(embed_word,embed_word,bias=True)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        #self.drop = nn.Dropout(prob)

    def forward(self,x_conv_out):
        """
        @param x_conv_out (Tensor): shape of input tensor [max_sentence_length, batch_size,e_word]
        @return x_emb(Tensor): shape [max_sentence_length, batch_size, e_word]
        """
        x_proj = self.relu(self.linear1(x_conv_out)) 
        x_gate = self.sig(self.linear2(x_conv_out))
        x_hightway = torch.mul(x_gate,x_proj) + torch.mul((1-x_gate),x_conv_out)
       # x_emb = self.drop(x_hightway)
       # return x_emb
        return x_hightway

### END YOUR CODE 

