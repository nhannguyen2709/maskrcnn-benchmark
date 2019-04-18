"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers, permute_and_flatten

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.layers import SoftCrossEntropyLoss, l2_bounded_regression_loss, l2_hint_loss
from maskrcnn_benchmark.layers.distil_loss import sigmoid_adapt_distill_loss_cpu
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation

class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0,
                 consistent=True):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm
        self.consistent = consistent

    def _refine_anchors(self, anchors, box_regression):
        anchors = list(zip(*anchors))
        refined_anchors = []
        for a, b in zip(anchors, box_regression):
            N, _, H, W = b.shape
            A = b.size(1) // 4
            b = permute_and_flatten(b, N, A, 4, H, W)
            b = b.reshape(N, -1, 4)
            results = []
            for per_anchors, per_box_regression in zip(a, b):
                per_refined_anchors = self.box_coder.decode(per_box_regression, per_anchors.bbox)
                per_boxlist = BoxList(per_refined_anchors, per_anchors.size, mode='xyxy')
                results.append(per_boxlist)
            refined_anchors.append(results)
        refined_anchors = list(zip(*refined_anchors))
        return refined_anchors

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        if self.consistent:
            refined_anchors = self._refine_anchors(anchors, box_regression)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)

        # consistent optimization loss
        if self.consistent:
            refined_anchors = [cat_boxlist(refined_anchors_per_image) 
                for refined_anchors_per_image in refined_anchors]
            refined_labels, refined_regression_targets = self.prepare_targets(refined_anchors, targets)
            refined_labels = torch.cat(refined_labels, dim=0)
            refined_regression_targets = torch.cat(refined_regression_targets, dim=0)
            refined_pos_inds = torch.nonzero(refined_labels > 0).squeeze(1)

            refined_regression_loss = smooth_l1_loss(
                box_regression[refined_pos_inds],
                refined_regression_targets[refined_pos_inds],
                beta=self.bbox_reg_beta,
                size_average=False,
            ) / (max(1, refined_pos_inds.numel() * self.regress_norm))

            refined_labels = refined_labels.int()

            refined_cls_loss = self.box_cls_loss_func(
                box_cls,
                refined_labels
            ) / (refined_pos_inds.numel() + N)
            return retinanet_cls_loss, retinanet_regression_loss, refined_cls_loss, refined_regression_loss

        return retinanet_cls_loss, retinanet_regression_loss


class RetinaNetDistilLossComputation(RetinaNetLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 soft_crossentropy_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.soft_distil_loss_func = soft_crossentropy_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, 
        adapted_student_features, teacher_features, 
        teacher_box_cls, teacher_box_regression, 
        targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        # teacher_anchors = self._refine_anchors(anchors, teacher_box_regression)
        # teacher_anchors = [cat_boxlist(teacher_anchors_per_image)
        #     for teacher_anchors_per_image in teacher_anchors]
        # teacher_labels, teacher_regression_targets = self.prepare_targets(teacher_anchors, targets)

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)
        teacher_box_cls, teacher_box_regression = \
                concat_box_prediction_layers(teacher_box_cls, teacher_box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        bg_inds = torch.nonzero(labels == 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)

        # soft_distil_loss = self.soft_distil_loss_func(
        #     box_cls, 
        #     teacher_box_cls, 
        #     bg_inds, 
        #     pos_inds
        #     ) / (bg_inds.numel() + pos_inds.numel() + N)
        soft_distil_loss = sigmoid_adapt_distill_loss_cpu(box_cls, 
            teacher_box_cls)
        teacher_bounded_regression_loss = l2_bounded_regression_loss(
            box_regression[pos_inds], 
            teacher_box_regression[pos_inds],
            regression_targets[pos_inds],
            size_average=False,
            ) / (max(1, pos_inds.numel() * self.regress_norm))
        hint_loss = l2_hint_loss(
            adapted_student_features, 
            teacher_features)

        mu = 0.5
        nu = 0.5
        gamma = 1.
        cls_loss = mu * retinanet_cls_loss + (1. - mu) * soft_distil_loss
        reg_loss = retinanet_regression_loss + nu * teacher_bounded_regression_loss
        hint_loss = gamma * hint_loss
        return cls_loss, reg_loss, hint_loss


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )
    soft_crossentropy_loss = SoftCrossEntropyLoss(

    )

    if cfg.MODEL.RETINANET.DISTIL_ON:
        loss_evaluator = RetinaNetDistilLossComputation(
            matcher,
            box_coder,
            generate_retinanet_labels,
            sigmoid_focal_loss,
            soft_crossentropy_loss,
            bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
            regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
        )
    else:
        loss_evaluator = RetinaNetLossComputation(
            matcher,
            box_coder,
            generate_retinanet_labels,
            sigmoid_focal_loss,
            bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
            regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
            consistent = cfg.MODEL.RETINANET.CONSISTENT,
        )
    return loss_evaluator
