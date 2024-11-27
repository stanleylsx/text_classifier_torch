# -*- coding: utf-8 -*-
# @Time : 2022/3/30 22:29
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : FastText.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from config import train_configure
import torch.nn.functional as F
import torch


class FastText(nn.Module, ABC):
    def __init__(self, vocab_size, num_labels):
        super(FastText, self).__init__()
        dropout_rate = train_configure['dropout_rate']
        embedding_dim = train_configure['embedding_dim']
        hidden_dim = train_configure['hidden_dim']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, inputs_embeds=None):
        if inputs_embeds is not None:
            embedding_output = inputs_embeds
        else:
            embedding_output = self.word_embeddings(x)
        output = embedding_output.mean(dim=1)
        output = self.fc1(output)
        output = F.relu(output)
        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            output = torch.mean(torch.stack([self.fc2(
                self.dropout(output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            output = self.dropout(output)
            output = self.fc2(output)
        if self.multilabel:
            prob = torch.sigmoid(output)
        else:
            prob = torch.softmax(output, dim=1)
        return output, prob
