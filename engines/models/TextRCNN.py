# -*- coding: utf-8 -*-
# @Time : 2022/3/30 22:29
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : TextRCNN.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from config import train_configure
import torch


class TextRCNN(nn.Module, ABC):
    def __init__(self, vocab_size, num_labels):
        super(TextRCNN, self).__init__()
        dropout_rate = train_configure['dropout_rate']
        embedding_dim = train_configure['embedding_dim']
        hidden_dim = train_configure['hidden_dim']
        max_sequence_length = train_configure['max_sequence_length']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.max_pool = nn.MaxPool1d(max_sequence_length)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, num_labels)

    def forward(self, x, inputs_embeds=None):
        if inputs_embeds is not None:
            embedding_output = inputs_embeds
        else:
            embedding_output = self.word_embeddings(x)
        rnn_output, _ = self.rnn(embedding_output)
        output = torch.cat((embedding_output, rnn_output), 2)
        output = output.permute(0, 2, 1)
        output = self.max_pool(output).squeeze(-1)
        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            output = torch.mean(torch.stack([self.fc(
                self.dropout(output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            dropout_output = self.dropout(output)
            output = self.fc(dropout_output)
        if self.multilabel:
            prob = torch.sigmoid(output)
        else:
            prob = torch.softmax(output, dim=1)
        return output, prob
