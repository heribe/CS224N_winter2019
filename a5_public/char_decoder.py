#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

from shutil import ignore_patterns
import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        v_char = len(target_vocab.char2id)
        # print("v_char:",v_char)
        self.charDecoder = nn.LSTM(char_embedding_size,hidden_size,bias=True)
        self.char_output_projection = nn.Linear(hidden_size,v_char,bias=True)
        if target_vocab!=None :
            self.decoderCharEmb = nn.Embedding(v_char,char_embedding_size,padding_idx=target_vocab.char2id['<pad>'])
            self.target_vocab = target_vocab


        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_embed = self.decoderCharEmb(input) # shape (wl,b,e_char)
        x_lstm,new_dec_hidden = self.charDecoder(input_embed,dec_hidden) # x_lstm shape(wl,b,h) #TODO dec_hidden是不是应该二维的
        s = self.char_output_projection(x_lstm) # shape(wl,b,V_char)

        return s,new_dec_hidden
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        lomm = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'],reduction='sum')
        s,_ = self.forward(char_sequence[:-1],dec_hidden) # (l-1,b,v_char)
        l,b,v_char = s.shape
        input = s.reshape(-1,v_char) # ((l-1)*b,v_char)
        target = char_sequence[1:].reshape(-1) # ((l-1)*b)
        loss = lomm(input,target)        
        # print(loss)
        return loss

        # scores,dec_hidden = self.forward(char_sequence[:-1],dec_hidden) #(length,batch_size,vocab_size)
        # entroy = nn.CrossEntropyLoss(ignore_index = self.target_vocab.char2id['<pad>'],reduction = 'sum') #不计算padding位置,将所有loss求和
        # loss = entroy(scores.permute(1,2,0),char_sequence[1:].permute(1,0)) 
        # #remove 'START' (batch_size,vocab_size,length) (batch_size,length) 第一维和最后一维要相同
        # # print(loss)
        # return loss
        
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        
        ### END YOUR CODE

