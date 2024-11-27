import torch.nn as nn


def init_network(model, method='xavier', exclude='embedding'):
    """
    权重初始化，默认xavier，不适用于预训练微调
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    if 'transformer' in name:
                        nn.init.uniform_(w, -0.1, 0.1)
                    else:
                        nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
    return model
