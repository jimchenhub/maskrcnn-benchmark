# Instance Relation
# author: lyujie chen
# date: 2019-06.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class RelationPostProcessor(nn.Module):
    '''
    '''

    def __init__(self):
        super(RelationPostProcessor, self).__init__()

    def forward(self):
        pass


def make_roi_relation_post_processor(cfg):
    relation_post_processor = RelationPostProcessor()
    return relation_post_processor
