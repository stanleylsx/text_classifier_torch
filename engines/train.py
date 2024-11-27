# -*- coding: utf-8 -*-
# @Time : 2021/7/13 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import torch
import pandas as pd
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from engines.utils.metrics import cal_single_label_metrics, cal_multi_label_metrics
from config import train_configure, distill_configure, stage, support_model
from torch import amp
from sklearn.model_selection import KFold


class Train:
    def __init__(self, data_manage, device, logger):
        self.logger = logger
        self.device = device
        self.data_manage = data_manage
        self.kfold = train_configure['kfold']
        self.multilabel = train_configure['multilabel']
        if stage == 'finetune':
            self.epoch = train_configure['f_epoch']
            self.learning_rate = train_configure['f_learning_rate']
            self.print_per_batch = train_configure['f_print_per_batch']
            self.is_early_stop = train_configure['f_is_early_stop']
            self.patient = train_configure['f_patient']
            self.batch_size = train_configure['f_batch_size']
            self.model_name = train_configure['f_model_name']
        elif stage == 'train_small_model':
            self.epoch = train_configure['s_epoch']
            self.learning_rate = train_configure['s_learning_rate']
            self.print_per_batch = train_configure['s_print_per_batch']
            self.is_early_stop = train_configure['s_is_early_stop']
            self.patient = train_configure['s_patient']
            self.batch_size = train_configure['s_batch_size']
            self.model_name = train_configure['s_model_name']
        else:
            self.epoch = distill_configure['epoch']
            self.learning_rate = distill_configure['learning_rate']
            self.print_per_batch = distill_configure['print_per_batch']
            self.is_early_stop = distill_configure['is_early_stop']
            self.patient = distill_configure['patient']
            self.batch_size = distill_configure['batch_size']
            self.student_model_name = distill_configure['student_model_name']
            self.teacher_model_name = distill_configure['teacher_model_name']
            self.alpha = distill_configure['alpha']
            self.temperature = float(distill_configure['temperature'])
            self.student_checkpoints_dir = distill_configure['checkpoints_dir']
        self.reverse_classes = data_manage.reverse_classes
        self.checkpoints_dir = train_configure['checkpoints_dir']

        self.optimizer = None
        self.gan = None

        if self.multilabel:
            if train_configure['use_r_drop']:
                raise ValueError('R-Drop is not compatible with multilabel train.')
            if train_configure['use_multilabel_categorical_cross_entropy']:
                from engines.utils.losses.multilabel_loss import MultilabelCategoricalCrossEntropy
                self.loss_function = MultilabelCategoricalCrossEntropy()
            else:
                from torch.nn import BCEWithLogitsLoss
                self.loss_function = BCEWithLogitsLoss()
        else:
            if train_configure['use_r_drop']:
                from engines.utils.losses.rdrop_loss import RDropLoss
                self.rdrop_loss_function = RDropLoss(device)

            if train_configure['use_poly_loss']:
                from engines.utils.losses.poly_loss import PolyLoss
                self.loss_function = PolyLoss(device)
            else:
                if train_configure['use_focal_loss']:
                    from engines.utils.losses.focal_loss import FocalLoss
                    self.loss_function = FocalLoss(device)
                else:
                    from torch.nn import CrossEntropyLoss
                    if train_configure['use_label_smoothing']:
                        smooth_factor = train_configure['smooth_factor']
                        self.loss_function = CrossEntropyLoss(label_smoothing=smooth_factor)
                    else:
                        self.loss_function = CrossEntropyLoss()

        self.use_fp16 = False
        if 'cuda' in str(device) and train_configure['use_fp16']:
            self.scaler = amp.GradScaler()
            self.use_fp16 = True

    def init_model(self):
        if stage == 'finetune':
            # 加载微调的预训练模型
            from engines.models.PretrainedModel import PretrainedModelClassification
            model_type = train_configure['f_model_type']
            model = PretrainedModelClassification(self.data_manage.num_labels, model_type=model_type).to(self.device)
            if train_configure['noisy_tune']:
                for name, para in model.named_parameters():
                    noise_lambda = train_configure['noise_lambda']
                    noise = (torch.rand(para.size()) - 0.5) * noise_lambda
                    noise = noise.to(self.device)
                    model.state_dict()[name][:] += noise * torch.std(para)
        elif stage == 'train_small_model':
            if train_configure['s_model_type'] == 'TextCNN':
                from engines.models.TextCNN import TextCNN
                model = TextCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'TextRNN':
                from engines.models.TextRNN import TextRNN
                model = TextRNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'TextRCNN':
                from engines.models.TextRCNN import TextRCNN
                model = TextRCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'FastText':
                from engines.models.FastText import FastText
                model = FastText(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif train_configure['s_model_type'] == 'Transformer':
                from engines.models.Transformer import Transformer
                model = Transformer(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            else:
                raise Exception('s_model_type does not exist')
        else:
            if distill_configure['student_model_type'] == 'TextCNN':
                from engines.models.TextCNN import TextCNN
                model = TextCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'TextRNN':
                from engines.models.TextRNN import TextRNN
                model = TextRNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'TextRCNN':
                from engines.models.TextRCNN import TextRCNN
                model = TextRCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'FastText':
                from engines.models.FastText import FastText
                model = FastText(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'Transformer':
                from engines.models.Transformer import Transformer
                model = Transformer(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif distill_configure['student_model_type'] == 'Bert':
                from engines.models.PretrainedModel import PretrainedModelClassification
                model_type = distill_configure['student_model_type']
                model = PretrainedModelClassification(self.data_manage.num_labels, model_type=model_type,
                                                      from_config=True).to(self.device)
            else:
                model = None

        if stage != 'finetune' and train_configure['init_network']:
            from engines.utils.init_network_parameter import init_network
            model = init_network(model, method=train_configure['init_network_method'])

        if train_configure['use_gan']:
            if train_configure['gan_method'] == 'fgm':
                from engines.utils.gan_utils import FGM
                self.gan = FGM(model)
            elif train_configure['gan_method'] == 'fgsm':
                from engines.utils.gan_utils import FGSM
                self.gan = FGSM(model)
            elif train_configure['gan_method'] == 'pgd':
                from engines.utils.gan_utils import PGD
                self.gan = PGD(model)
            elif train_configure['gan_method'] == 'freelb':
                from engines.utils.gan_utils import FreeLB
                self.gan = FreeLB(model)
            elif train_configure['gan_method'] == 'awp':
                from engines.utils.gan_utils import AWP
                self.gan = AWP(model)
            else:
                self.gan = None

        params = list(model.parameters())
        optimizer_type = train_configure['optimizer']
        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, lr=self.learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=self.learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise Exception('optimizer_type does not exist')
        return model

    def split_data(self):
        validation_rate = train_configure['validation_rate']
        train_file = train_configure['train_file']
        val_file = train_configure['val_file']
        train_data = pd.read_csv(train_file, encoding='utf-8', usecols=['label', 'sentence'])[['label', 'sentence']]
        if train_configure['multilabel']:
            train_data['label'] = train_data.label.apply(self.data_manage.map_multilabel)
        else:
            train_data = train_data.loc[train_data.label.isin(self.data_manage.classes)]
            train_data['label'] = train_data.label.map(lambda x: self.data_manage.class_id[x])

        if val_file == '':
            self.logger.info('generate validation dataset...')
            ratio = 1 - validation_rate
            train_data, val_data = train_data[:int(ratio * len(train_data))], train_data[int(ratio * len(train_data)):]
        else:
            val_data = pd.read_csv(val_file, encoding='utf-8', usecols=['label', 'sentence'])[['label', 'sentence']]
            if train_configure['multilabel']:
                val_data['label'] = val_data.label.apply(self.data_manage.map_multilabel)
            else:
                val_data = val_data.loc[val_data.label.isin(self.data_manage.classes)]
                val_data['label'] = val_data.label.map(lambda x: self.data_manage.class_id[x])
        train_loader = DataLoader(dataset=train_data.values,
                                  batch_size=self.batch_size,
                                  collate_fn=self.data_manage.get_dataset,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data.values,
                                collate_fn=self.data_manage.get_dataset,
                                batch_size=self.batch_size)
        self.logger.info('train dataset nums:{}'.format(len(train_data)))
        self.logger.info('validation dataset nums:{}'.format(len(val_data)))
        return train_loader, val_loader

    def train(self):
        if self.kfold:
            kfold_start_time = time.time()
            train_file = train_configure['train_file']
            fold_splits = train_configure['fold_splits']
            train_df = pd.read_csv(train_file, encoding='utf-8')
            kf = KFold(n_splits=fold_splits, random_state=2, shuffle=True)
            fold = 1
            for train_index, val_index in kf.split(train_df):
                self.logger.info(f'\nTraining fold {fold}...\n')
                model = self.init_model()
                self.optimizer.zero_grad()
                train_data = train_df.loc[train_index]
                val_data = train_df.loc[val_index]
                train_dataset = self.data_manage.get_dataset(train_data)
                val_dataset = self.data_manage.get_dataset(val_data)
                train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size)
                self.train_each_fold(model, train_loader, val_loader, fold_index=fold)
                fold = fold + 1
            self.logger.info('\nKfold: total training time consumption: %.3f(min)' % ((time.time() - kfold_start_time) / 60))
        else:
            model = self.init_model()
            if os.path.exists(os.path.join(self.checkpoints_dir, self.model_name)):
                self.logger.info('Resuming from checkpoint...')
                model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
                optimizer_checkpoint = torch.load(os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
                self.optimizer.load_state_dict(optimizer_checkpoint['optimizer'])
            else:
                self.logger.info('Initializing from scratch.')
            train_loader, val_loader = self.split_data()
            self.train_each_fold(model, train_loader, val_loader)

    def train_each_fold(self, model, train_loader, val_loader, fold_index=None):
        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        step_total = self.epoch * len(train_loader)
        global_step = 0
        scheduler = None
        model_name = self.model_name + '_' + str(fold_index) if fold_index else self.model_name

        if train_configure['warmup']:
            scheduler_type = train_configure['scheduler_type']
            if train_configure['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = train_configure['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        if train_configure['ema']:
            from engines.utils.ema import EMA
            ema = EMA(model)
            ema.register()
        else:
            ema = None

        if train_configure['swa']:
            from torch.optim.swa_utils import AveragedModel, SWALR
            model = AveragedModel(model).to(self.device)
            swa_lr = train_configure['swa_lr']
            anneal_epochs = train_configure['anneal_epochs']
            swa_scheduler = SWALR(optimizer=self.optimizer, swa_lr=swa_lr, anneal_epochs=anneal_epochs,
                                  anneal_strategy='linear')
        else:
            swa_scheduler = None

        very_start_time = time.time()
        self.logger.info(('+' * 20) + 'training starting' + ('+' * 20))
        for i in range(self.epoch):
            start_time = time.time()
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            model.train()
            step = 0
            for batch in tqdm(train_loader):
                token_id, labels = batch
                token_id = token_id.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                if self.use_fp16:
                    with amp.autocast('cuda'):
                        logits, _ = model(token_id)
                        if train_configure['use_r_drop']:
                            logits_2, _ = model(token_id)
                            loss = self.rdrop_loss_function(logits, logits_2, labels)
                        else:
                            loss = self.loss_function(logits, labels)
                        self.scaler.scale(loss).backward()
                        if train_configure['use_gan']:
                            k = train_configure['attack_round']
                            if train_configure['gan_method'] in ('fgm', 'fgsm'):
                                self.gan.attack()
                                logits, _ = model(token_id)
                                if train_configure['use_r_drop']:
                                    logits_2, _ = model(token_id)
                                    loss = self.rdrop_loss_function(logits, logits_2, labels)
                                else:
                                    loss = self.loss_function(logits, labels)
                                self.scaler.scale(loss).backward()
                                self.gan.restore()  # 恢复embedding参数
                            elif train_configure['gan_method'] == 'pgd':
                                self.gan.backup_grad()
                                for t in range(k):
                                    self.gan.attack(is_first_attack=(t == 0))
                                    if t != k - 1:
                                        model.zero_grad()
                                    else:
                                        self.gan.restore_grad()
                                    logits, _ = model(token_id)
                                    if train_configure['use_r_drop']:
                                        logits_2, _ = model(token_id)
                                        loss = self.rdrop_loss_function(logits, logits_2, labels)
                                    else:
                                        loss = self.loss_function(logits, labels)
                                    self.scaler.scale(loss).backward()
                                self.gan.restore()
                            elif train_configure['gan_method'] == 'freelb':
                                delta = self.gan.init_delta(token_id)
                                embeds_init = self.gan.get_embeddings(token_id)
                                for t in range(k):
                                    delta.requires_grad_()
                                    inputs_embeds = delta + embeds_init
                                    logits, _ = model(token_id, inputs_embeds=inputs_embeds)
                                    if train_configure['use_r_drop']:
                                        logits_2, _ = model(token_id, inputs_embeds)
                                        loss = self.rdrop_loss_function(logits, logits_2, labels)
                                    else:
                                        loss = self.loss_function(logits, labels)
                                    loss = loss / self.gan.gradient_accumulation_steps
                                    self.scaler.scale(loss).backward()
                                    delta_grad = delta.grad.clone().detach()
                                    if self.gan.adv_norm_type == 'linf':
                                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1,
                                                            p=float('inf')).view(-1, 1, 1)
                                        denorm = torch.clamp(denorm, min=1e-8)
                                        delta = (delta + self.gan.adv_lr * delta_grad / denorm).detach()
                                        if self.gan.adv_max_norm > 0:
                                            delta = torch.clamp(delta, -self.gan.adv_max_norm,
                                                                self.gan.adv_max_norm).detach()
                                    else:
                                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1,
                                                                                                                 1)
                                        denorm = torch.clamp(denorm, min=1e-8)
                                        delta = (delta + self.gan.adv_lr * delta_grad / denorm).detach()
                                        if self.gan.adv_max_norm > 0:
                                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2,
                                                                    dim=1).detach()
                                            exceed_mask = (delta_norm > self.gan.adv_max_norm).to(embeds_init)
                                            reweights = (self.gan.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                                            delta = (delta * reweights).detach()
                                    embeds_init = self.gan.get_embeddings(token_id)
                            elif train_configure['gan_method'] == 'awp':
                                if i + 1 >= self.gan.awp_start:
                                    self.gan.attack_backward()
                                    logits, _ = model(token_id)
                                    if train_configure['use_r_drop']:
                                        logits_2, _ = model(token_id, inputs_embeds)
                                        loss = self.rdrop_loss_function(logits, logits_2, labels)
                                    else:
                                        loss = self.loss_function(logits, labels)
                                    self.optimizer.zero_grad()
                                    self.scaler.scale(loss).backward()
                                    self.gan.restore()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    logits, _ = model(token_id)
                    if train_configure['use_r_drop']:
                        logits_2, _ = model(token_id)
                        loss = self.rdrop_loss_function(logits, logits_2, labels)
                    else:
                        loss = self.loss_function(logits, labels)
                    loss.backward()
                    if train_configure['use_gan']:
                        k = train_configure['attack_round']
                        if train_configure['gan_method'] in ('fgm', 'fgsm'):
                            self.gan.attack()
                            logits, _ = model(token_id)
                            if train_configure['use_r_drop']:
                                logits_2, _ = model(token_id)
                                loss = self.rdrop_loss_function(logits, logits_2, labels)
                            else:
                                loss = self.loss_function(logits, labels)
                            loss.backward()
                            self.gan.restore()  # 恢复embedding参数
                        elif train_configure['gan_method'] == 'pgd':
                            self.gan.backup_grad()
                            for t in range(k):
                                self.gan.attack(is_first_attack=(t == 0))
                                if t != k - 1:
                                    model.zero_grad()
                                else:
                                    self.gan.restore_grad()
                                logits, _ = model(token_id)
                                if train_configure['use_r_drop']:
                                    logits_2, _ = model(token_id)
                                    loss = self.rdrop_loss_function(logits, logits_2, labels)
                                else:
                                    loss = self.loss_function(logits, labels)
                                loss.backward()
                            self.gan.restore()
                        elif train_configure['gan_method'] == 'freelb':
                            delta = self.gan.init_delta(token_id)
                            embeds_init = self.gan.get_embeddings(token_id)
                            for t in range(k):
                                delta.requires_grad_()
                                inputs_embeds = delta + embeds_init
                                logits, _ = model(token_id, inputs_embeds=inputs_embeds)
                                if train_configure['use_r_drop']:
                                    logits_2, _ = model(token_id, inputs_embeds)
                                    loss = self.rdrop_loss_function(logits, logits_2, labels)
                                else:
                                    loss = self.loss_function(logits, labels)
                                loss = loss / self.gan.gradient_accumulation_steps
                                loss.backward()
                                delta_grad = delta.grad.clone().detach()
                                if self.gan.adv_norm_type == 'linf':
                                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1,
                                                        p=float('inf')).view(-1, 1, 1)
                                    denorm = torch.clamp(denorm, min=1e-8)
                                    delta = (delta + self.gan.adv_lr * delta_grad / denorm).detach()
                                    if self.gan.adv_max_norm > 0:
                                        delta = torch.clamp(delta, -self.gan.adv_max_norm,
                                                            self.gan.adv_max_norm).detach()
                                else:
                                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                                    denorm = torch.clamp(denorm, min=1e-8)
                                    delta = (delta + self.gan.adv_lr * delta_grad / denorm).detach()
                                    if self.gan.adv_max_norm > 0:
                                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2,
                                                                dim=1).detach()
                                        exceed_mask = (delta_norm > self.gan.adv_max_norm).to(embeds_init)
                                        reweights = (self.gan.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                                        delta = (delta * reweights).detach()
                                embeds_init = self.gan.get_embeddings(token_id)
                        elif train_configure['gan_method'] == 'awp':
                            if i + 1 >= self.gan.awp_start:
                                self.gan.attack_backward()
                                logits, _ = model(token_id)
                                if train_configure['use_r_drop']:
                                    logits_2, _ = model(token_id, inputs_embeds)
                                    loss = self.rdrop_loss_function(logits, logits_2, labels)
                                else:
                                    loss = self.loss_function(logits, labels)
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.gan.restore()
                self.optimizer.step()

                if train_configure['ema']:
                    ema.update()

                if train_configure['swa']:
                    if global_step > train_configure['swa_start_step']:
                        model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        if train_configure['warmup']:
                            scheduler.step()

                if train_configure['warmup']:
                    scheduler.step()

                if step % self.print_per_batch == 0 and step != 0:
                    if self.multilabel:
                        predictions = torch.sigmoid(logits)
                        predictions = torch.where(predictions > 0.5, 1, 0)
                        measures = cal_multi_label_metrics(y_true=labels.to('cpu').numpy(), y_pred=predictions.to('cpu').numpy())
                    else:
                        predictions = torch.argmax(logits, dim=-1)
                        measures, _ = cal_single_label_metrics(y_true=labels.to('cpu').numpy(), y_pred=predictions.to('cpu').numpy())
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (step, loss, res_str))
                step = step + 1
                global_step = global_step + 1

            if train_configure['ema']:
                ema.apply_shadow()

            measures = self.validate(model, val_loader)

            if train_configure['ema']:
                ema.restore()

            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)

            if measures['f1'] > best_f1_val:
                unprocessed = 0
                best_f1_val = measures['f1']
                best_at_epoch = i + 1
                if train_configure['swa']:
                    torch.optim.swa_utils.update_bn(train_loader, model, device=self.device)
                if not self.kfold:
                    optimizer_checkpoint = {'optimizer': self.optimizer.state_dict()}
                    torch.save(optimizer_checkpoint, os.path.join(self.checkpoints_dir, model_name + '.optimizer'))
                torch.save(model.state_dict(), os.path.join(self.checkpoints_dir, model_name))
                self.logger.info('saved the new best model with f1: %.3f' % best_f1_val)
            else:
                unprocessed += 1

            if self.is_early_stop:
                if unprocessed >= self.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' %
                                     ((time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, model, val_loader):
        with torch.no_grad():
            model.eval()
            self.logger.info('start evaluate engines...')
            loss_values = []
            y_true, y_pred = np.array([]), np.array([])
            first = True
            for batch in tqdm(val_loader):
                if stage == 'distillation':
                    if distill_configure['teacher_model_type'] in support_model['pretrained_types'] and \
                            distill_configure['student_model_type'] not in support_model['pretrained_types']:
                        _, token_id, labels = batch
                    else:
                        token_id, labels = batch
                else:
                    token_id, labels = batch
                token_id = token_id.to(self.device)
                labels = labels.to(self.device)
                if stage == 'distillation' and distill_configure['self_distillation']:
                    logits, _, _ = model(token_id)
                else:
                    logits, _ = model(token_id)
                loss = self.loss_function(logits, labels)
                loss_values.append(loss.item())
                if self.multilabel:
                    predictions = torch.sigmoid(logits)
                    predictions = torch.where(predictions > 0.5, 1, 0)
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

        if self.multilabel:
            measures = cal_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        else:
            measures, each_classes = cal_single_label_metrics(y_true=y_true, y_pred=y_pred)
            # 打印每一个类别的指标
            classes_val_str = ''
            for k, _ in each_classes.items():
                try:
                    map_k = str(int(float(k)))
                except ValueError:
                    continue
                if map_k in self.reverse_classes:
                    classes_val_str += (self.reverse_classes[map_k] + ': ' + str(each_classes[k]) + '\n')
            self.logger.info(classes_val_str)
        # 打印损失函数
        val_res_str = 'loss: %.3f ' % np.mean(loss_values)
        for k, _ in measures.items():
            val_res_str += (k + ': %.3f ' % measures[k])
        self.logger.info(val_res_str)
        return measures

    def init_teacher_model(self):
        model_type = distill_configure['teacher_model_type']
        if model_type in support_model['pretrained_types']:
            from engines.models.PretrainedModel import PretrainedModelClassification
            teacher_model = PretrainedModelClassification(
                self.data_manage.num_labels, model_type=model_type).to(self.device)
        else:
            if model_type == 'TextCNN':
                from engines.models.TextCNN import TextCNN
                teacher_model = TextCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif model_type == 'TextRNN':
                from engines.models.TextRNN import TextRNN
                teacher_model = TextRNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif model_type == 'TextRCNN':
                from engines.models.TextRCNN import TextRCNN
                teacher_model = TextRCNN(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif model_type == 'FastText':
                from engines.models.FastText import FastText
                teacher_model = FastText(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            elif model_type == 'Transformer':
                from engines.models.Transformer import Transformer
                teacher_model = Transformer(self.data_manage.vocab_size, self.data_manage.num_labels).to(self.device)
            else:
                raise Exception('teacher_model_type does not exist')
        return teacher_model

    def distill_each_model(self, student_model, teacher_model, train_loader, val_loader):
        from torch.nn import CrossEntropyLoss
        from engines.utils.losses.distill_loss import DistillLoss
        ce_loss_function = CrossEntropyLoss()
        distill_loss_function = DistillLoss()
        very_start_time = time.time()
        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0

        step_total = self.epoch * len(train_loader)
        global_step = 0
        scheduler = None

        if train_configure['warmup']:
            scheduler_type = train_configure['scheduler_type']
            if train_configure['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = train_configure['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        self.logger.info(('+' * 20) + 'distilling starting' + ('+' * 20))
        for i in range(self.epoch):
            start_time = time.time()
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            student_model.train()
            losses = []
            step = 0
            for batch in tqdm(train_loader):
                if distill_configure['teacher_model_type'] in support_model['pretrained_types'] and \
                        distill_configure['student_model_type'] not in support_model['pretrained_types']:
                    teacher_token_id, student_X, labels = batch
                    teacher_token_id = teacher_token_id.to(self.device)
                    student_X = student_X.to(self.device)
                    labels = labels.to(self.device)
                else:
                    teacher_token_id, labels = batch
                    teacher_token_id = teacher_token_id.to(self.device)
                    student_X = teacher_token_id
                    labels = labels.to(self.device)
                teacher_logits, _ = teacher_model(teacher_token_id)
                student_logits, _ = student_model(student_X)
                ce_loss = ce_loss_function(student_logits, labels)
                distill_loss = distill_loss_function.cal_loss(student_logits, teacher_logits)
                merge_loss = self.alpha * ce_loss + (1 - self.alpha) * distill_loss
                merge_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(merge_loss.item())

                if train_configure['warmup']:
                    scheduler.step()

                if step % self.print_per_batch == 0 and step != 0:
                    predictions = torch.argmax(student_logits, dim=-1)
                    measures, _ = cal_single_label_metrics(y_true=labels.to('cpu').numpy(), y_pred=predictions.to('cpu').numpy())
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (step, ce_loss, res_str))
                step = step + 1
                global_step = global_step + 1

            measures = self.validate(student_model, val_loader)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)
            if measures['f1'] > best_f1_val:
                unprocessed = 0
                best_f1_val = measures['f1']
                best_at_epoch = i + 1
                torch.save(student_model.state_dict(), os.path.join(
                    self.student_checkpoints_dir, self.student_model_name))
                self.logger.info('saved the new best model with f1: %.3f' % best_f1_val)
            else:
                unprocessed += 1

            if self.is_early_stop:
                if unprocessed >= self.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info(
                        'total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        self.logger.info('total distilling time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    @staticmethod
    def simple_adaptor(batch, model_outputs):
        logits, prob, hidden_states, attention_states = model_outputs
        return {'logits': logits, 'hidden': hidden_states, 'attention': attention_states}

    def distillation(self):
        if self.kfold:
            student_model = self.init_model()
            train_loader, val_loader = self.split_data()
            very_start_time = time.time()
            for fold_index in range(train_configure['fold_splits']):
                self.logger.info(f'\ndistilling fold {fold_index + 1}')
                fold_model_name = self.teacher_model_name + '_' + str(fold_index + 1)
                teacher_model = self.init_teacher_model()
                teacher_model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, fold_model_name)))
                teacher_model.eval()
                self.distill_each_model(student_model, teacher_model, train_loader, val_loader)
            self.logger.info(
                'Kfold: total distilling time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
        else:
            teacher_model = self.init_teacher_model()
            teacher_model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.teacher_model_name)))
            teacher_model.eval()

            if distill_configure['self_distillation']:
                from engines.textbrewer import GeneralDistiller
                from engines.textbrewer import TrainingConfig, DistillationConfig
                train_loader, val_loader = self.split_data()
                student_model = self.init_model()
                num_training_steps = len(train_loader) * self.epoch
                scheduler_type = train_configure['scheduler_type']

                if train_configure['num_warmup_steps'] == -1:
                    num_warmup_steps = num_training_steps * 0.1
                else:
                    num_warmup_steps = train_configure['num_warmup_steps']

                if scheduler_type == 'linear':
                    from transformers import get_linear_schedule_with_warmup
                    scheduler_class = get_linear_schedule_with_warmup
                    scheduler_args = {'num_warmup_steps': num_warmup_steps,
                                      'num_training_steps': num_training_steps}
                elif scheduler_type == 'cosine':
                    from transformers import get_cosine_schedule_with_warmup
                    scheduler_class = get_cosine_schedule_with_warmup
                    scheduler_args = {'num_warmup_steps': num_warmup_steps,
                                      'num_training_steps': num_training_steps}
                else:
                    raise Exception('scheduler_type does not exist')
                distill_config = DistillationConfig(
                    temperature=distill_configure['temperature'],
                    kd_loss_type=distill_configure['distillation_method'],
                    intermediate_matches=distill_configure['intermediate_matches'])
                train_config = TrainingConfig(output_dir=self.student_checkpoints_dir,
                                              model_name=self.student_model_name, device=self.device,
                                              ckpt_frequency=1, ckpt_epoch_frequency=1)
                distiller = GeneralDistiller(logger=self.logger,
                                             train_config=train_config, distill_config=distill_config,
                                             model_T=teacher_model, model_S=student_model,
                                             adaptor_T=self.simple_adaptor, adaptor_S=self.simple_adaptor)
                with distiller:
                    distiller.train(self.optimizer, train_loader, self.print_per_batch, val_loader, self.epoch,
                                    scheduler_class=scheduler_class,
                                    scheduler_args=scheduler_args, callback=self.validate)
            else:
                train_loader, val_loader = self.split_data()
                student_model = self.init_model()
                if os.path.exists(os.path.join(distill_configure['checkpoints_dir'],
                                               distill_configure['student_model_name'])):
                    self.logger.info('Resuming from checkpoint...')
                    student_model.load_state_dict(torch.load(os.path.join(distill_configure['checkpoints_dir'],
                                                                          distill_configure['student_model_name'])))
                else:
                    self.logger.info('Initializing from scratch.')
                self.distill_each_model(student_model, teacher_model, train_loader, val_loader)

    def prune(self):
        from engines.textpruner import GeneralConfig, VocabularyPruningConfig, TransformerPruningConfig
        from engines.textpruner import VocabularyPruner, TransformerPruner, PipelinePruner
        from engines.models.PretrainedModel import PretrainedModelClassification
        from config import prune_configure
        general_config = GeneralConfig(use_device=self.device, output_dir=prune_configure['checkpoints_dir'])
        train_file, val_file = train_configure['train_file'], train_configure['val_file']
        data_df = pd.read_csv(train_file, encoding='utf-8')
        data_list = data_df.sentence.tolist()
        if val_file != '':
            dev_df = pd.read_csv(val_file, encoding='utf-8')
            data_df = pd.concat([data_df, dev_df], axis=0)
            data_list.extend(dev_df.sentence.tolist())

        model_type = prune_configure['model_type']
        tokenizer = self.data_manage.tokenizer_for_pretrained_model(model_type)
        if prune_configure['from_config']:
            model = PretrainedModelClassification(self.data_manage.num_labels, model_type=model_type,
                                                  from_config=True).to(self.device)
        else:
            model = PretrainedModelClassification(self.data_manage.num_labels, model_type=model_type).to(self.device)
        model.load_state_dict(torch.load(os.path.join(prune_configure['teacher_checkpoint_dir'],
                                                      prune_configure['teacher_model_name'])))
        model.eval()

        vocabulary_pruning_config = VocabularyPruningConfig(min_count=prune_configure['min_count'],
                                                            prune_lm_head='auto')
        transformer_pruning_config = TransformerPruningConfig(
            target_ffn_size=prune_configure['intermediate_size'],
            target_num_of_heads=prune_configure['num_attention_heads'],
            pruning_method='iterative',
            n_iters=prune_configure['iters'])

        if prune_configure['where'] == 'vocabulary':
            pruner = VocabularyPruner(model, tokenizer, vocabulary_pruning_config=vocabulary_pruning_config,
                                      general_config=general_config, base_model_prefix=model_type)
            pruner.prune(dataiter=data_list, save_model=True)
        elif prune_configure['where'] == 'transformer':
            dataset = self.data_manage.get_dataset(data_df)
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size)
            pruner = TransformerPruner(model, transformer_pruning_config=transformer_pruning_config,
                                       general_config=general_config)
            pruner.prune(dataloader=data_loader, save_model=True)
        else:
            dataset = self.data_manage.get_dataset(data_df)
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size)
            pruner2 = PipelinePruner(model, tokenizer, transformer_pruning_config=transformer_pruning_config,
                                     vocabulary_pruning_config=vocabulary_pruning_config,
                                     general_config=general_config)
            pruner2.prune(dataloader=data_loader, dataiter=data_list, save_model=True)
