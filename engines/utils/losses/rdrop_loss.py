from config import train_configure
import torch.nn.functional as F
import torch


class RDropLoss(torch.nn.Module):
    """
    r-drop loss
    """
    def __init__(self, device):
        super(RDropLoss, self).__init__()
        self.alpha = 4
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

    @staticmethod
    def compute_kl_loss(p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(self, p, q, y_true):
        loss_1 = self.loss_function(p, y_true)
        loss_2 = self.loss_function(q, y_true)
        ce_loss = 0.5 * (loss_1 + loss_2)
        kl_loss = self.compute_kl_loss(p, q)
        loss = ce_loss + self.alpha * kl_loss
        return loss
