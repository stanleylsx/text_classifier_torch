# -*- coding: utf-8 -*-
# @Time : 2022/3/15 22:29
# @Author : Stanley
# @EMail : gzlishouxian@gmail.com
# @File : Transformer.py
# @Software: PyCharm
from torch import nn
from config import train_configure
import torch
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(Transformer, self).__init__()
        dropout_rate = train_configure['dropout_rate']
        embedding_dim = train_configure['embedding_dim']
        hidden_dim = train_configure['hidden_dim']
        sequence_length = train_configure['max_sequence_length']
        encoder_num = train_configure['encoder_num']
        head_num = train_configure['head_num']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=head_num, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num)
        self.pos_encoder = PositionalEncoding(embedding_dim, sequence_length)
        self.avg_pool = nn.AvgPool1d
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, num_labels)

    def forward(self, input_ids, inputs_embeds=None):
        if inputs_embeds is not None:
            out = inputs_embeds
        else:
            out = self.word_embeddings(input_ids)
        out = self.pos_encoder(out)
        out = self.encoder(out)
        out = F.avg_pool1d(out.transpose(1, 2), out.size(1)).squeeze(-1)
        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            out = torch.mean(torch.stack([self.fc(
                self.dropout(out)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            out = self.dropout(out)
            out = self.fc(out)
        if self.multilabel:
            prob = torch.sigmoid(out)
        else:
            prob = torch.softmax(out, dim=1)
        return out, prob


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, seq_length):
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        pe = np.array([[pos / np.power(10000, 2 * (i // 2) / embedding_dim)
                        for i in range(embedding_dim)] for pos in range(seq_length)])
        # 偶数列使用sin，奇数列使用cos
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = torch.from_numpy(pe).to(torch.float32)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 在这里将词的embedding和位置embedding相加
        position_embed = x + self.pe
        return position_embed
