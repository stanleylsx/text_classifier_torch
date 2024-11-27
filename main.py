# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.train import Train
from engines.predict import Predictor
from config import train_configure, distill_configure, prune_configure, stage, mode, use_cuda, cuda_device
from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.utils.setup_seed import setup_seed
import json
import os
import torch


def fold_check(config, name):

    if config['checkpoints_dir'] == '':
        raise Exception(name + ':checkpoints_dir did not set...')

    if name == 'train' and stage != 'finetune' and not os.path.exists(os.path.dirname(config['token_file'])):
        print(name + ':token fold not found, creating...')
        os.makedirs(os.path.dirname(config['token_file']))

    if not os.path.exists(config['checkpoints_dir']):
        print(name + ':checkpoints fold not found, creating...')
        os.makedirs(config['checkpoints_dir'])

    if not os.path.exists(config['checkpoints_dir'] + '/logs'):
        print('log fold not found, creating...')
        os.mkdir(config['checkpoints_dir'] + '/logs')


if __name__ == '__main__':
    setup_seed(train_configure['seed'])
    if stage == 'distillation':
        fold_check(config=train_configure, name='train')
        fold_check(config=distill_configure, name='distillation')
        logger = get_logger(distill_configure['checkpoints_dir'] + '/logs', mode)
    elif stage == 'prune':
        fold_check(config=prune_configure, name='prune')
        logger = get_logger(prune_configure['checkpoints_dir'] + '/logs', mode)
    else:
        fold_check(config=train_configure, name='train')
        logger = get_logger(train_configure['checkpoints_dir'] + '/logs', mode)

    if use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda_device}')
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                " Make sure CUDA is available or set use_cuda=False."
            )
    else:
        device = 'cpu'
    logger.info(f'device: {device}')

    data_manage = DataManager(logger)
    # 训练分类器
    if mode == 'train_classifier':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        train = Train(data_manage, device, logger)
        if stage == 'distillation':
            logger.info('stage: distillation')
            logger.info(json.dumps(distill_configure, indent=2, ensure_ascii=False))
            train.distillation()
        elif stage == 'prune':
            logger.info('stage: prune')
            logger.info(json.dumps(prune_configure, indent=2, ensure_ascii=False))
            train.prune()
        else:
            logger.info('stage: train')
            train.train()
    elif mode == 'interactive_predict':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        logger.info(json.dumps(distill_configure, indent=2, ensure_ascii=False))
        logger.info('mode: predict_one')
        logger.info('stage: {}'.format(stage))
        predictor = Predictor(data_manage, device, logger)
        predictor.predict_one('warm start')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            logger.info('input:{}'.format(str(sentence)))
            result = predictor.predict_one(sentence)
            logger.info('output:{}'.format(str(result)))
            print(result)
    elif mode == 'test':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        logger.info(json.dumps(distill_configure, indent=2, ensure_ascii=False))
        logger.info('mode: test')
        logger.info('stage: {}'.format(stage))
        predictor = Predictor(data_manage, device, logger)
        predictor.predict_test()
    elif mode == 'convert_onnx':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        logger.info(json.dumps(distill_configure, indent=2, ensure_ascii=False))
        logger.info('stage: {}'.format(stage))
        logger.info('mode: convert_onnx')
        predictor = Predictor(data_manage, device, logger)
        predictor.convert_onnx()
    elif mode == 'convert_torch_script':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        logger.info(json.dumps(distill_configure, indent=2, ensure_ascii=False))
        logger.info('stage: {}'.format(stage))
        logger.info('mode: convert_torch_script')
        predictor = Predictor(data_manage, device, logger)
        predictor.convert_torch_script()
    elif mode == 'show_model_info':
        logger.info(json.dumps(train_configure, indent=2, ensure_ascii=False))
        logger.info('mode: show_model_info')
        predictor = Predictor(data_manage, device, logger)
        predictor.show_model_info()
