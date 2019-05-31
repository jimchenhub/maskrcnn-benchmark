# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import ranking_loss


class RelationLossComputation(object):
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, labels, pred, gt):
        pass


def make_roi_relation_loss_evaluator(cfg):
    loss_weight = cfg.MODEL.RELATION_LOSS_WEIGHT
    loss_evaluator = RelationLossComputation(loss_weight)

    return loss_evaluator


