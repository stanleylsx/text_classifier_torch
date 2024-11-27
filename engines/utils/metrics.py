# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from config import train_configure
import numpy as np


def cal_single_label_metrics(y_true, y_pred):
    """
    单标签指标计算
    """
    average = train_configure['metrics_average']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    each_classes = classification_report(y_true, y_pred, output_dict=True, labels=np.unique(y_pred), zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}, each_classes


def cal_multi_label_metrics(y_true, y_pred):
    """
    多标签指标计算
    """
    average = train_configure['metrics_average']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
