# Instance Relation
# author: lyujie chen
# date: 2019-06.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


class RelationPostProcessor(nn.Module):
    '''
    '''

    def __init__(self):
        super(RelationPostProcessor, self).__init__()
        self.masker = Masker(threshold=0.5, padding=1)

    def forward(self, x, boxes):
        pos = 0
        for box in boxes:
            rel_val = x[pos:pos+len(box)]
            rel_val = torch.squeeze(rel_val, dim=1)
            pos += len(box)
            box.add_field("relation_val", rel_val)
            
            # refine mask result based on relation value
            masks = box.get_field("mask")
            masks = self.masker([masks], [box])[0].cpu().numpy()
            new_masks = masks.copy()
            num = len(masks)
            for i in range(num):
                for j in range(i + 1, num):
                    overlap = cv2.bitwise_and(masks[i][0], masks[j][0])
                    is_overlap = overlap.any()
                    if is_overlap:
                        if rel_val[i] < rel_val[j]:
                            new_masks[j] = new_masks[j] - overlap
                        elif rel_val[j] < rel_val[i]:
                            new_masks[i] = new_masks[i] - overlap
            box.add_field("mask", new_masks)

        return boxes

def make_roi_relation_post_processor(cfg):
    relation_post_processor = RelationPostProcessor()
    return relation_post_processor
