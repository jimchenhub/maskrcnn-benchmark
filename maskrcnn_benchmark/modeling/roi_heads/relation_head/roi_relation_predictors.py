# Instance Relation
# author: lyujie chen
# date: 2019-06.

from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class RelationPredictor(nn.Module):
    def __init__(self, cfg):
        super(RelationPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.maskiou = nn.Linear(1024, num_classes)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)


    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou


def make_roi_relation_predictor(cfg):
    func = RelationPredictor
    return func(cfg)
