# -*- coding: utf-8 -*-
# @Time : 2021/7/16 22:29
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : TextRNN.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from config import train_configure
import torch


class TextRNN(nn.Module, ABC):
    def __init__(self, vocab_size, num_labels):
        super(TextRNN, self).__init__()
        dropout_rate = train_configure['dropout_rate']
        embedding_dim = train_configure['embedding_dim']
        hidden_dim = train_configure['hidden_dim']

        self.use_attention = train_configure['use_attention']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        if self.use_attention:
            self.W_w = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
            self.u_w = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
            nn.init.uniform_(self.W_w, -0.1, 0.1)
            nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x, inputs_embeds=None):
        if inputs_embeds is not None:
            embedding_output = inputs_embeds
        else:
            embedding_output = self.word_embeddings(x)
        rnn_output, _ = self.rnn(embedding_output)

        if self.use_attention:
            score = torch.tanh(torch.matmul(rnn_output, self.W_w))
            attention_weights = torch.softmax(torch.matmul(score, self.u_w), dim=1)
            rnn_output = rnn_output * attention_weights

        rnn_output = torch.sum(rnn_output, dim=1)
        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            outputs = torch.mean(torch.stack([self.fc(
                self.dropout(rnn_output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            dropout_output = self.dropout(rnn_output)
            outputs = self.fc(dropout_output)
        if self.multilabel:
            prob = torch.sigmoid(outputs)
        else:
            prob = torch.softmax(outputs, dim=1)
        return outputs, prob
