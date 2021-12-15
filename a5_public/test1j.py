import torch
import torch.nn as nn
from vocab import VocabEntry
from model_embeddings import ModelEmbeddings

mes = ModelEmbeddings(256,VocabEntry())
input = torch.randint(91,(10,3,21))
output = mes(input)
print(output.shape)
