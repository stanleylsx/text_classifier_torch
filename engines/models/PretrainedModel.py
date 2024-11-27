# -*- coding: utf-8 -*-
# @Time : 2022/6/14 21:54
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : PretrainedModel.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from config import train_configure, stage, distill_configure, prune_configure, mode
import torch


class PretrainedModelClassification(nn.Module, ABC):
    def __init__(self, num_labels, model_type, from_config=False):
        super().__init__()
        self.num_labels = num_labels

        if not from_config:
            if model_type == 'Bert':
                from transformers import BertModel
                ptm = train_configure['ptm']
                self.model = BertModel.from_pretrained(ptm)

            elif model_type == 'DistilBert':
                from transformers import DistilBertModel
                ptm = train_configure['ptm']
                self.model = DistilBertModel.from_pretrained(ptm)

            elif model_type == 'RoBerta':
                from transformers import RobertaModel
                ptm = train_configure['ptm']
                self.model = RobertaModel.from_pretrained(ptm)

            elif model_type == 'ALBert':
                from transformers import AlbertModel
                ptm = train_configure['ptm']
                self.model = AlbertModel.from_pretrained(ptm)

            elif model_type == 'XLNet':
                from transformers import XLNetModel
                ptm = train_configure['ptm']
                self.model = XLNetModel.from_pretrained(ptm)

            elif model_type == 'Electra':
                from transformers import ElectraModel
                ptm = train_configure['ptm']
                self.model = ElectraModel.from_pretrained(ptm)

            elif model_type in ['MiniLM', 'DeBertaV3', 'XLM-RoBERTa']:
                from transformers import AutoModel
                ptm = train_configure['ptm']
                self.model = AutoModel.from_pretrained(ptm)
        else:
            if model_type == 'Bert':
                from transformers import BertModel, BertConfig
                if stage == 'distillation':
                    bert_config = BertConfig.from_dict(distill_configure['distill_mlm_config'])
                elif stage == 'prune':
                    bert_config = BertConfig.from_json_file(prune_configure['checkpoints_dir'] + '/' + 'config.json')
                else:
                    bert_config = {}
                self.model = BertModel(bert_config)

        if distill_configure['self_distillation']:
            self.self_distillation = True
            self.model.config.output_attentions = True
        else:
            self.self_distillation = False

        self.config = self.model.config
        hidden_size = self.config.hidden_size
        dropout_rate = train_configure['dropout_rate']
        self.multilabel = train_configure['multilabel']
        self.multisample_dropout = train_configure['multisample_dropout']
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, inputs_embeds=None):
        input_mask = torch.where(input_ids > 0, 1, 0)
        if inputs_embeds is not None:
            model_output = self.model(inputs_embeds=inputs_embeds, attention_mask=input_mask, output_hidden_states=True)
        else:
            model_output = self.model(input_ids, attention_mask=input_mask, output_hidden_states=True)
        last_hidden_state = model_output.last_hidden_state
        hidden_states = model_output.hidden_states
        attention_states = model_output.attentions
        pooled_output = last_hidden_state[:, 0]
        if self.multisample_dropout and train_configure['dropout_round'] > 1:
            dropout_round = train_configure['dropout_round']
            logits = torch.mean(torch.stack([self.fc(
                self.dropout(pooled_output)) for _ in range(dropout_round)], dim=0), dim=0)
        else:
            pooled_output = self.dropout(pooled_output)
            logits = self.fc(pooled_output)
        if self.multilabel:
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=1)
        if mode == 'train_classifier' and stage == 'distillation' and self.self_distillation:
            return logits, prob, hidden_states, attention_states
        else:
            return logits, prob
