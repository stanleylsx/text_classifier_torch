# -*- coding: utf-8 -*-
# @Time : 2022/8/30 23:32
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : distill_loss.py
# @Software: PyCharm
import torch
from config import distill_configure


class DistillLoss(torch.nn.Module):
    def __init__(self):
        super(DistillLoss, self).__init__()
        self.temperature = distill_configure['temperature']
        self.distillation_method = distill_configure['distillation_method']

        if self.distillation_method == 'kl':
            from torch.nn import KLDivLoss
            self.distill_loss = KLDivLoss(reduction='batchmean')
        elif self.distillation_method == 'mse':
            from torch.nn import MSELoss
            self.distill_loss = MSELoss()

    def temperature_softmax(self, logits):
        return torch.softmax(logits / self.temperature, dim=-1)

    def temperature_log_softmax(self, logits):
        return torch.log_softmax(logits / self.temperature, dim=-1)

    def cal_loss(self, student_logits, teacher_logits):
        if self.distillation_method == 'ce':
            pt = self.temperature_softmax(teacher_logits)
            lpt = self.temperature_log_softmax(student_logits)
            loss = -(pt * lpt).sum(dim=-1).mean()
            return loss
        else:
            return self.distill_loss(self.temperature_log_softmax(student_logits),
                                     self.temperature_softmax(teacher_logits))
