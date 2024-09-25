import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class BCEFocalLoss(torch.nn.Module):
    # """
    # input: [N, C]
    # target: [N, ]
    # """
  def __init__(self, gamma=2, alpha=0.25, reduction='mean',device_now = "cuda: 0"):
    super(BCEFocalLoss, self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction
    self.device_now = device_now

  def forward(self, predict, target):
    pt = torch.sigmoid(predict)[:,1:]
    target = target.unsqueeze(-1).float()

    loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)


    if self.reduction == 'mean':
        loss = torch.mean(loss)
    elif self.reduction == 'sum':
        loss = torch.sum(loss)
    return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean',device_now = "cuda: 0"):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num
        self.device_now = device_now

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha.to(self.device_now) * (torch.pow((1 - probs.to(self.device_now)), self.gamma)) * log_p.to(self.device_now)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    

    


def log_sum_exp(x):
    b, _ = torch.max(x, 1)
    b= torch.unsqueeze(torch.squeeze(b),dim=1)
    y = b + torch.log(torch.exp(x - b.repeat(1,x.size(1))).sum(1,True))
    # y.size() = [N, 1]. Squeeze to [N] and return
    return y.squeeze(1)


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)



def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1

    loss = log_sum_exp(logits) - class_select(logits, target)

    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean',device_now = "cuda: 0"):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.device_now = device_now

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
            # return MultiCEFocalLoss(class_num=7, reduction='sum', device_now = self.device_now)(input, target.long())
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
            # return MultiCEFocalLoss(class_num=7, reduction='mean', device_now = self.device_now)(input, target.long())
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)