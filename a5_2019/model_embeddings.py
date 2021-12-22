#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        e_char = 50
        prob = 0.3
        self.embed_size = embed_size
        #e_word = 256
        self.embedding = nn.Embedding(len(vocab.char2id),e_char,padding_idx=vocab.char2id['<pad>'])
        self.drop = nn.Dropout(prob)
        self.cnn = CNN(e_char,embed_size)
        self.highway = Highway(embed_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        
        char_es = self.embedding(input) # shape (sl,b,wl,e_char)
        char_es_shape = char_es.transpose(2,3) # shape (sl,b,e_char,wl)
        char_batch = char_es_shape.reshape(-1,char_es_shape.shape[-2],char_es_shape.shape[-1]) # shape (sl*b,e_char,wl)
        x_cnn = self.cnn(char_batch) #shape (sl*b,e_word)
        x_cnn_shape = x_cnn.reshape(char_es_shape.shape[0],char_es_shape.shape[1],-1)        
        x_emb = self.highway(x_cnn_shape) # shape (sl,b,e_word)
        x_emb_drop = self.drop(x_emb)
        return x_emb_drop
        ### END YOUR CODE

