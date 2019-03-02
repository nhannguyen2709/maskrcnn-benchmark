"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
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
                 regress_norm=1.0):
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

    @torch.no_grad()
    def __obtain_refined_anchors(self, anchors, box_cls, box_regression, targets):
        N = len(anchors)

        refined_anchors = [] # (list[BoxList])
        reshaped_box_regression = box_regression.reshape(N, -1, 4)
        for box_regression_per_image, anchors_per_image in zip(
            reshaped_box_regression, anchors):
            refined_anchors_per_image = self.box_coder.decode(box_regression_per_image, anchors_per_image.bbox)
            boxlist = BoxList(refined_anchors_per_image, anchors_per_image.size, mode='xyxy')
            refined_anchors.append(boxlist)
        refined_labels, refined_regression_targets = self.prepare_targets(refined_anchors, targets)
            
        return refined_labels, refined_regression_targets

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

        # Consistency optimization loss here
        refined_labels, refined_regression_targets = self.__obtain_refined_anchors(anchors, box_cls, box_regression, targets)
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

        return retinanet_cls_loss, retinanet_regression_loss, refined_regression_loss, refined_cls_loss


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

    loss_evaluator = RetinaNetLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
    )
    return loss_evaluator
