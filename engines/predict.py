# -*- coding: utf-8 -*-
# @Time : 2021/7/19 23:32
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import torch
import time
import os
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import DataLoader
from config import stage, train_configure, distill_configure, prune_configure
from tqdm import tqdm
from engines.utils.metrics import cal_single_label_metrics, cal_multi_label_metrics
from texttable import Texttable
from collections import Counter


class Predictor:
    def __init__(self, data_manager, device, logger):
        self.device = device
        self.logger = logger
        self.data_manager = data_manager
        self.num_labels = data_manager.num_labels
        self.vocab_size = data_manager.vocab_size
        self.map_multilabel = data_manager.map_multilabel
        self.kfold = train_configure['kfold']
        self.multilabel = train_configure['multilabel']
        self.reverse_classes = data_manager.reverse_classes

        if stage == 'finetune':
            model_name = train_configure['f_model_name']
            self.checkpoints_dir = train_configure['checkpoints_dir']
        elif stage == 'train_small_model':
            model_name = train_configure['s_model_name']
            self.checkpoints_dir = train_configure['checkpoints_dir']
        elif stage == 'prune':
            model_name = 'pytorch_model.bin'
            self.checkpoints_dir = prune_configure['checkpoints_dir']
        else:
            model_name = distill_configure['student_model_name']
            self.checkpoints_dir = distill_configure['checkpoints_dir']

        if self.kfold and stage != 'distillation':
            self.model_list = []
            for fold_index in range(train_configure['fold_splits']):
                fold_model_name = model_name + '_' + str(fold_index + 1)
                model = self.init_model()
                model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, fold_model_name)))
                model.eval()
                self.model_list.append(model)
        else:
            self.model = self.init_model()
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, model_name)))
            self.model.eval()

    def init_model(self):
        if stage == 'finetune':
            # 加载微调的预训练模型
            from engines.models.PretrainedModel import PretrainedModelClassification
            model_type = train_configure['f_model_type']
            model = PretrainedModelClassification(self.num_labels, model_type=model_type).to(self.device)
        elif stage == 'train_small_model':
            # 加载单独小模型
            if train_configure['s_model_type'] == 'TextCNN':
                from engines.models.TextCNN import TextCNN
                model = TextCNN(self.vocab_size, self.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'TextRNN':
                from engines.models.TextRNN import TextRNN
                model = TextRNN(self.vocab_size, self.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'TextRCNN':
                from engines.models.TextRCNN import TextRCNN
                model = TextRCNN(self.vocab_size, self.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'FastText':
                from engines.models.FastText import FastText
                model = FastText(self.vocab_size, self.num_labels).to(self.device)
            else:
                from engines.models.Transformer import Transformer
                model = Transformer(self.vocab_size, self.num_labels).to(self.device)
        elif stage == 'prune':
            from engines.models.PretrainedModel import PretrainedModelClassification
            model_type = prune_configure['model_type']
            model = PretrainedModelClassification(self.data_manager.num_labels, model_type=model_type,
                                                  from_config=True).to(self.device)
        else:
            # 加载蒸馏后的模型
            if distill_configure['student_model_type'] == 'TextCNN':
                from engines.models.TextCNN import TextCNN
                model = TextCNN(self.vocab_size, self.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'TextRNN':
                from engines.models.TextRNN import TextRNN
                model = TextRNN(self.vocab_size, self.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'TextRCNN':
                from engines.models.TextRCNN import TextRCNN
                model = TextRCNN(self.vocab_size, self.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'FastText':
                from engines.models.FastText import FastText
                model = FastText(self.vocab_size, self.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'Transformer':
                from engines.models.Transformer import Transformer
                model = Transformer(self.vocab_size, self.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'Bert':
                from engines.models.PretrainedModel import PretrainedModelClassification
                model_type = distill_configure['student_model_type']
                model = PretrainedModelClassification(self.num_labels, model_type=model_type,
                                                      from_config=True).to(self.device)
            else:
                model = None
        return model

    def predict_one(self, sentence):
        start_time = time.time()
        token = self.data_manager.prepare_single_sentence(sentence)
        token = token.to(self.device)
        if self.kfold and stage != 'distillation':
            model_outputs_list = []
            for model in self.model_list:
                model_outputs, _ = model(token)
                model_outputs = list(model_outputs.to('cpu').tolist()[0])
                model_outputs_list.append(model_outputs)
            merge_result = torch.tensor(np.mean(model_outputs_list, axis=0))
            if self.multilabel:
                probability = torch.sigmoid(merge_result)
                probability = probability.squeeze(0)
                indices = torch.nonzero(probability >= 0.5).flatten().to('cpu').tolist()
                prediction_label = [self.data_manager.classes[index] for index in indices]
                this_probability = {self.data_manager.classes[index]: float(probability[index]) for index in indices}
            else:
                prediction = int(torch.argmax(merge_result, dim=-1))
                probability = torch.softmax(merge_result, dim=0)
                this_probability = list(probability.tolist()[0])[prediction]
                prediction_label = self.reverse_classes[str(prediction)]
        else:
            model_outputs, probability = self.model(token)
            if self.multilabel:
                probability = probability.squeeze(0)
                indices = torch.nonzero(probability >= 0.5).flatten().to('cpu').tolist()
                prediction_label = [self.data_manager.classes[index] for index in indices]
                this_probability = {self.data_manager.classes[index]: float(probability[index]) for index in indices}
            else:
                prediction = torch.argmax(model_outputs.to('cpu'), dim=-1).numpy()[0]
                this_probability = list(probability.tolist()[0])[prediction]
                prediction_label = self.reverse_classes[str(prediction)]
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        return prediction_label, this_probability

    def predict_test(self):
        test_file = train_configure['test_file']
        if test_file == '':
            self.logger.info('test dataset does not exist!')
            return
        test_data = pd.read_csv(test_file, encoding='utf-8', usecols=['label', 'sentence'])[['label', 'sentence']]
        if train_configure['multilabel']:
            test_data['label'] = test_data.label.apply(self.map_multilabel)
        else:
            test_data = test_data.loc[test_data.label.isin(self.data_manager.classes)]
            test_data['label'] = test_data.label.map(lambda x: self.data_manager.class_id[x])
        batch_size = train_configure['f_batch_size']
        test_loader = DataLoader(dataset=test_data.values,
                                 collate_fn=self.data_manager.get_dataset,
                                 batch_size=batch_size)
        y_true, y_pred, probabilities = np.array([]), np.array([]), np.array([])
        first = True
        start_time = time.time()
        for batch in tqdm(test_loader):
            token_id, labels = batch
            token_id = token_id.to(self.device)
            labels = labels.to(self.device)
            if self.kfold and stage != 'distillation':
                logits_list = []
                for model in self.model_list:
                    logits, _ = model(token_id)
                    model_outputs = list(logits.to('cpu').tolist())
                    logits_list.append(model_outputs)
                merge_result = torch.tensor(np.mean(logits_list, axis=0))
                if self.multilabel:
                    probability = torch.sigmoid(merge_result)
                    predictions = torch.where(probability > 0.5, 1, 0)
                    if first:
                        y_true = labels.to('cpu')
                        y_pred = predictions.to('cpu')
                        first = False
                        continue
                    y_true = np.concatenate([y_true, labels.to('cpu')], axis=0)
                    y_pred = np.concatenate([y_pred, predictions.to('cpu')], axis=0)
                else:
                    y_true = np.append(y_true, labels.to('cpu'))
                    y_pred = np.append(y_pred, torch.argmax(merge_result, dim=1))
                    probability = torch.softmax(merge_result, dim=1)
                    max_probability = torch.max(probability, dim=1).values.detach().numpy()
                    probabilities = np.append(probabilities, max_probability)
            else:
                logits, probability = self.model(token_id)
                if self.multilabel:
                    predictions = torch.where(probability > 0.5, 1, 0)
                    if first:
                        y_true = labels.to('cpu')
                        y_pred = predictions.to('cpu')
                        first = False
                        continue
                    y_true = np.concatenate([y_true, labels.to('cpu')], axis=0)
                    y_pred = np.concatenate([y_pred, predictions.to('cpu')], axis=0)
                else:
                    predictions = torch.argmax(logits, dim=-1)
                    y_true = np.append(y_true, labels.to('cpu'))
                    y_pred = np.append(y_pred, predictions.to('cpu'))
                    max_probability = torch.max(probability.to('cpu'), dim=1).values.detach().numpy()
                    probabilities = np.append(probabilities, max_probability)

        self.logger.info('test time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        if self.multilabel:
            measures = cal_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        else:
            measures, each_classes = cal_single_label_metrics(y_true=y_true, y_pred=y_pred)
            count_map = {self.reverse_classes[str(int(key))]: value for key, value in Counter(y_true).items()}
            # 打印不一致的下标
            inconsistent = np.argwhere(y_true != y_pred)
            if len(inconsistent) > 0:
                indices = [i for i in list(inconsistent.ravel())]
                y_error_pred = [self.reverse_classes[str(int(i))] for i in list(y_pred[indices])]
                data_dict = {'indices': indices, 'y_error_pred': y_error_pred}
                for col_name in test_data.columns.values.tolist():
                    data_dict[col_name] = test_data.iloc[indices][col_name].tolist()
                data_dict['probability'] = list(probabilities[indices])
                indices = [i + 1 for i in indices]
                test_result_file = self.checkpoints_dir + '/logs/' + (datetime.datetime.now().strftime('%Y%m%d%H%M%S-badcase.csv'))
                result = pd.DataFrame(data_dict)
                result['label'] = result.label.apply(lambda x: self.reverse_classes[str(x)])
                statics = pd.DataFrame(
                    result.groupby('label').apply(lambda data: data.y_error_pred.value_counts())).reset_index()
                statics['error_rate'] = statics.apply(lambda row: row['count'] / count_map[row['label']], axis=1)
                tb = Texttable()
                tb.set_cols_align(['l', 'r', 'r', 'r'])
                tb.set_cols_dtype(['t', 't', 'i', 'f'])
                tb.header(list(statics.columns))
                tb.add_rows(statics.values, header=False)
                result.to_csv(test_result_file, encoding='utf-8', index=False)
                self.logger.info('\nerror indices in test dataset:')
                self.logger.info(indices)
                self.logger.info('\nerror distribution:')
                self.logger.info(tb.draw())
            # 打印每一个类别的指标
            self.logger.info('\nevery classes result:')
            classes_val_str = ''
            for k, v in each_classes.items():
                try:
                    map_k = str(int(float(k)))
                except ValueError:
                    continue
                if map_k in self.data_manager.reverse_classes:
                    classes_val_str += ('\n' + self.data_manager.reverse_classes[map_k] + ': ' + str(each_classes[k]))
            self.logger.info(classes_val_str)

        # 打印总的指标
        res_str = ''
        self.logger.info('\nglobal result:')
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        self.logger.info('%s' % res_str)

    def convert_onnx(self):
        max_sequence_length = self.data_manager.max_sequence_length
        dummy_input = torch.ones([1, max_sequence_length]).to(self.device).int()
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model, dummy_input, f=onnx_path, opset_version=14,
                          input_names=['input'], output_names=['logits', 'probability'], do_constant_folding=True,
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'},
                                        'probability': {0: 'batch_size'}})
        self.logger.info('convert torch to onnx successful...')

    def convert_torch_script(self):
        max_sequence_length = self.data_manager.max_sequence_length
        example_input = torch.ones([1, max_sequence_length]).to(self.device).int()
        traced_script_module = torch.jit.trace(self.model, example_input)
        torch_script_path = os.path.join(self.checkpoints_dir, 'model.pt')
        traced_script_module.save(torch_script_path)
        del traced_script_module
        self.logger.info('convert to Torch Script successful...')

    def show_model_info(self):
        from engines.textpruner import summary
        info = summary(self.model, max_level=3)
        self.logger.info(info)
