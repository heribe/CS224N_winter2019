#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
from torch import nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,e_char,e_word,m_word=21,k=5):
        # Use a kernel size of k = 5
        # ses one nn.Conv1d layer
        k=5
        super(CNN,self).__init__()
        self.conv1d = nn.Conv1d(e_char,e_word,k)
        self.pool = nn.MaxPool1d(m_word-k+1)    

    def forward(self,x_reshaped):
        """
        @param x_reshaped(tensor): shape [max_sentence_length, batch_size,char_embedding, max_word_length]
        @return x_conv_out(tensor): shape [max_sentence_length, batch_size, word_embedding]
        """
        xconv = F.relu(self.conv1d(x_reshaped))
        x_conv_out = self.pool(xconv)
        x_conv_out = torch.squeeze(x_conv_out,dim=-1)
        return x_conv_out

### END YOUR CODE
