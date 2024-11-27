# -*- coding: utf-8 -*-
# @Time : 2021/7/16 22:29
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : TextCNN.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from config import train_configure
import torch.nn.functional as F
import torch


class TextCNN(nn.Module, ABC):
    def __init__(self, vocab_size, num_labels):
        super(TextCNN, self).__init__()
        dropout_rate = train_configure['dropout_rate']
        embedding_dim = train_configure['embedding_dim']
        num_filters = train_configure['num_filters']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv2d(1, num_filters, (2, embedding_dim))
        self.conv2 = nn.Conv2d(1, num_filters, (3, embedding_dim))
        self.conv3 = nn.Conv2d(1, num_filters, (4, embedding_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * 3, num_labels)

    def forward(self, x, inputs_embeds=None):
        if inputs_embeds is not None:
            embedding_output = inputs_embeds
        else:
            embedding_output = self.word_embeddings(x)
        inputs = torch.unsqueeze(embedding_output, 1)
        conv1 = self.conv1(inputs)
        conv1 = torch.relu(conv1).squeeze(3)
        pool1 = F.max_pool1d(conv1, int(conv1.size(2))).squeeze(2)

        conv2 = self.conv2(inputs)
        conv2 = torch.relu(conv2).squeeze(3)
        pool2 = F.max_pool1d(conv2, int(conv2.size(2))).squeeze(2)

        conv3 = self.conv3(inputs)
        conv3 = torch.relu(conv3).squeeze(3)
        pool3 = F.max_pool1d(conv3, int(conv3.size(2))).squeeze(2)

        concat_outputs = torch.cat((pool1, pool2, pool3), 1)
        concat_outputs = concat_outputs.squeeze(-1).squeeze(-1)

        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            outputs = torch.mean(torch.stack([self.fc(
                self.dropout(concat_outputs)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            dropout_outputs = self.dropout(concat_outputs)
            outputs = self.fc(dropout_outputs)
        if self.multilabel:
            prob = torch.sigmoid(outputs)
        else:
            prob = torch.softmax(outputs, dim=1)
        return outputs, prob
