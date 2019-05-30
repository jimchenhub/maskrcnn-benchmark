# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator


class ROIRelationHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_relation_feature_extractor(cfg)
        self.predictor = make_roi_relation_predictor(cfg)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

    def forward(self, bbox_features, mask_features, labels, relation_targets):
        """
        Arguments:


        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            results (list[BoxList]): during training, returns None. During testing, the predicted boxlists are returned.
                with the `mask` field set
        """
        pass


def build_roi_relation_head(cfg):
    return ROIRelationHead(cfg)
