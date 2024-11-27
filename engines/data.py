# -*- coding: utf-8 -*-
# @Time : 2021/07/12 22:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from tqdm import tqdm
from engines.utils.clean_data import filter_word, filter_char
from collections import Counter
from config import train_configure, stage, distill_configure, prune_configure, mode, support_model
import pandas as pd
import jieba
import torch
import os


class DataManager:

    def __init__(self, logger):
        self.logger = logger
        self.max_sequence_length = train_configure['max_sequence_length']
        self.train_file = train_configure['train_file']
        self.val_file = train_configure['val_file']
        self.token_file = train_configure['token_file']
        self.token_level = train_configure['token_level']

        if train_configure['stop_words'] and os.path.exists(train_configure['stop_words_file']):
            self.stop_words_file = train_configure['stop_words_file']
            self.stop_words = self.get_stop_words()
        else:
            self.stop_words = []
        self.remove_sp = True if train_configure['remove_special'] else False

        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'

        assert self.max_sequence_length <= 512, '超过序列最大长度设定'

        if stage == 'finetune':
            model_type = train_configure['f_model_type']
            self.tokenizer = self.tokenizer_for_pretrained_model(model_type)
            self.vocab_size = len(self.tokenizer)
        elif stage == 'distillation':
            model_type = distill_configure['teacher_model_type']
            self.tokenizer = self.tokenizer_for_pretrained_model(model_type)
            self.vocab_size = len(self.tokenizer)
            if not distill_configure['self_distillation']:
                self.token2id, self.id2token = self.load_vocab()
                self.vocab_size = len(self.token2id) + 1
        elif stage == 'prune':
            model_type = prune_configure['model_type']
            self.token_config_path = prune_configure['checkpoints_dir']
            if mode != 'train_classifier':
                self.tokenizer = self.tokenizer_for_pretrained_model(model_type, from_config=True)
            else:
                self.tokenizer = self.tokenizer_for_pretrained_model(model_type)
            self.vocab_size = len(self.tokenizer)
        else:
            self.token2id, self.id2token = self.load_vocab()
            self.vocab_size = len(self.token2id) + 1
        self.classes = train_configure['classes']
        self.num_labels = len(self.classes)
        self.class_id = {cls: index for index, cls in enumerate(self.classes)}
        self.reverse_classes = {str(class_id): class_name for class_name, class_id in self.class_id.items()}

    @staticmethod
    def processing_sentence(x):
        cut_word = jieba.cut(str(x).strip())
        words = list(cut_word)
        words = [word for word in words if word != ' ']
        return words

    def get_stop_words(self):
        stop_words_list = []
        try:
            with open(self.stop_words_file, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list

    def load_vocab(self):
        if not os.path.exists(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            df = pd.read_csv(self.train_file, encoding='utf-8')
            if not self.val_file == '':
                val_df = pd.read_csv(self.val_file).sample(frac=1)
                df = pd.concat([df, val_df], axis=0)
            return self.build_vocab(df['sentence'])
        token2id, id2token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        return token2id, id2token

    def build_vocab(self, sentences):
        tokens = []
        if self.token_level == 'word':
            # 词粒度
            for sentence in tqdm(sentences):
                words = self.processing_sentence(sentence)
                if self.stop_words:
                    words = [word for word in words if word not in self.stop_words and word != ' ']
                tokens.extend(words)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 2 and filter_word(k)]
        else:
            # 字粒度
            for sentence in tqdm(sentences):
                chars = list(str(sentence))
                if self.stop_words:
                    chars = [char for char in chars if char not in self.stop_words and char != ' ']
                tokens.extend(chars)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if filter_char(k, remove_sp=self.remove_sp)]
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        # 向生成的词表和标签表中加入[PAD]
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(id2token)] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(id2token)
        # 保存词表及标签表
        with open(self.token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token

    def padding(self, token, pretrained_model=False):
        pad_token_id = self.tokenizer.pad_token_id if pretrained_model else 0
        if len(token) < self.max_sequence_length:
            token += [pad_token_id for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
        return token

    def tokenizer_for_pretrained_model(self, model_type, from_config=False):
        if not from_config:
            if model_type == 'Bert':
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type == 'DistilBert':
                from transformers import DistilBertTokenizer
                tokenizer = DistilBertTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type == 'RoBerta':
                from transformers import RobertaTokenizer
                tokenizer = RobertaTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type == 'ALBert':
                from transformers import AlbertTokenizer
                tokenizer = AlbertTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type == 'XLNet':
                from transformers import XLNetTokenizer
                tokenizer = XLNetTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type == 'Electra':
                from transformers import ElectraTokenizer
                tokenizer = ElectraTokenizer.from_pretrained(train_configure['ptm'])
            elif model_type in ['MiniLM', 'DeBertaV3', 'XLM-RoBERTa']:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(train_configure['ptm'])
            else:
                tokenizer = None
        else:
            if model_type == 'Bert':
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(self.token_config_path)
            else:
                tokenizer = None
        return tokenizer

    def tokenizer_for_sentences(self, sent):
        tokens = []
        for token in sent:
            if token in self.token2id:
                tokens.append(self.token2id[token])
            else:
                tokens.append(self.token2id[self.UNKNOWN])
        tokens = self.padding(tokens)
        return tokens

    def prepare_pretrained_model_data(self, data):
        sentences, labels = [], []
        for label, sentence in data:
            labels.append(label)
            sentences.append(sentence)
        token_results = self.tokenizer.batch_encode_plus(sentences, padding=True, truncation=True,
                                                         max_length=self.max_sequence_length, return_tensors='pt')
        token_ids = token_results.get('input_ids')
        return token_ids, torch.tensor(labels)

    def prepare_data(self, data):
        sentences, labels = [], []
        max_lens = 0
        for label, sentence in data:
            labels.append(label)
            if self.token_level == 'word':
                sentence = self.processing_sentence(sentence)
            else:
                sentence = list(sentence)
            if self.stop_words:
                sentence = [word for word in sentence if word not in self.stop_words and word != ' ']
            tokens = self.tokenizer_for_sentences(sentence)
            lens = len(tokens)
            max_lens = lens if lens > max_lens else max_lens
            sentences.append(torch.tensor(tokens))
        max_lens = self.max_sequence_length if max_lens > self.max_sequence_length else max_lens
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
        sentences = sentences[:, :max_lens]
        return sentences, torch.tensor(labels)

    def get_dataset(self, data):
        """
        构建Dataset
        """
        # convert the data in matrix
        if stage == 'distillation':
            if mode == 'test':
                if distill_configure['student_model_type'] in support_model['pretrained_types']:
                    student_token_id, labels = self.prepare_pretrained_model_data(data)
                else:
                    student_token_id, labels = self.prepare_data(data)
                return student_token_id, labels
            else:
                if distill_configure['teacher_model_type'] in support_model['pretrained_types']:
                    if distill_configure['student_model_type'] in support_model['pretrained_types']:
                        student_token_id, labels = self.prepare_pretrained_model_data(data)
                        return student_token_id, labels
                    else:
                        student_token_id, labels = self.prepare_data(data)
                        teacher_token_id, _ = self.prepare_pretrained_model_data(data)
                        return teacher_token_id, student_token_id, labels
                else:
                    student_token_id, labels = self.prepare_data(data)
                    return student_token_id, labels
        elif stage in ['finetune', 'prune']:
            token_id, labels = self.prepare_pretrained_model_data(data)
            return token_id, labels
        else:
            token_id, labels = self.prepare_data(data)
            return token_id, labels

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        if stage in ['finetune', 'prune'] or (stage == 'distillation' and distill_configure['student_model_type'] in support_model['pretrained_types']):
            token_results = self.tokenizer(sentence)
            token_ids = self.padding(token_results.get('input_ids'), pretrained_model=True)
            return torch.tensor([token_ids])
        else:
            if self.token_level == 'word':
                sentence = self.processing_sentence(sentence)
            else:
                sentence = list(sentence)
            if self.stop_words:
                sentence = [word for word in sentence if word not in self.stop_words and word != ' ']
            tokens = self.tokenizer_for_sentences(sentence)
            return torch.tensor([tokens])

    def map_multilabel(self, labels):
        label = [0] * self.num_labels
        for x in labels.split(','):
            if x in self.classes:
                label[int(self.class_id[x])] = 1
        return label
