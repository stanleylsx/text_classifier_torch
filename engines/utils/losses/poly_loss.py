from config import train_configure
import torch.nn.functional as F
import torch


class PolyLoss(torch.nn.Module):
    def __init__(self, device, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.classes = train_configure['classes']
        self.epsilon = epsilon
        self.num_labels = len(self.classes)
        if train_configure['use_focal_loss']:
            from engines.utils.losses.focal_loss import FocalLoss
            self.loss_function = FocalLoss(device, reduction='none')
        else:
            from torch.nn import CrossEntropyLoss
            if train_configure['use_label_smoothing']:
                smooth_factor = train_configure['smooth_factor']
                self.loss_function = CrossEntropyLoss(label_smoothing=smooth_factor)
            else:
                self.loss_function = CrossEntropyLoss()

    def forward(self, inputs, targets):
        if train_configure['use_focal_loss']:
            p = torch.sigmoid(inputs)
            labels = F.one_hot(targets, self.num_labels).float()
            poly = labels * p + (1 - labels) * (1 - p)
            focal_loss = self.loss_function(inputs, targets)
            loss = self.epsilon * torch.pow(1 - poly, self.loss_function.gamma + 1)
            loss = focal_loss + torch.mean(loss, dim=-1)
        else:
            poly = torch.sum(F.one_hot(targets, self.num_labels).float() * F.softmax(inputs, dim=1), dim=-1)
            loss = self.loss_function(inputs, targets)
            loss = loss + self.epsilon * (1 - poly)
        loss = torch.mean(loss, dim=0)
        return loss
