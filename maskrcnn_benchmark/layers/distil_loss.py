import torch
import torch.nn as nn


class SoftCrossEntropyLoss(nn.Module):
    """
    Compute the cross-entropy between student and teacher
    probability distributions.
    """
    def __init__(self, bg_weight=1.5, fg_weight=1.):
        super(SoftCrossEntropyLoss, self).__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight

    def forward(self, student_logits, teacher_logits,
        bg_inds, fg_inds):
        p_s = torch.sigmoid(student_logits)
        p_t = torch.sigmoid(teacher_logits)
        bg_loss = -self.bg_weight * p_t[bg_inds] * torch.log(p_s[bg_inds])
        fg_loss = -self.fg_weight * p_t[fg_inds] * torch.log(p_s[fg_inds])
        return bg_loss.sum() + fg_loss.sum()


def l2_bounded_regression_loss(student_reg, teacher_reg, target, m=0.5, size_average=True):
    d_s = torch.dist(student_reg, target, p=2)
    d_t = torch.dist(teacher_reg, target, p=2)
    cond = (d_s + m) > d_t
    loss = torch.where(cond, d_s, d_s * 0.)
    if size_average:
        return loss.mean()
    return loss.sum()


def l2_hint_loss(student_features, teacher_features):
    loss = 0.
    for x_s, x_t in zip(student_features, teacher_features):
        loss += torch.dist(x_s, x_t, p=2) / x_s[0].numel()
    loss *= x_s.size(0)
    return loss