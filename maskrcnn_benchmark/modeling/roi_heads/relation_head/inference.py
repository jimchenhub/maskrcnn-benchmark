# Instance Relation
# author: lyujie chen
# date: 2019-06.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# import cv2
# from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
# import math
# import time


class RelationPostProcessor(nn.Module):
    '''
    '''
    def __init__(self):
        super(RelationPostProcessor, self).__init__()
        # self.masker = Masker(threshold=0.5, padding=1)

    def forward(self, x, boxes):
        pos = 0

        # t1 = time.time()
        for box in boxes:
            rel_val = x[pos:pos+len(box)]
            rel_val = torch.squeeze(rel_val, dim=1)
            pos += len(box)
            box.add_field("relation_val", rel_val)

            # # refine mask result based on relation value
            # masks = box.get_field("mask")
            # masks = self.masker([masks], [box])[0].detach()
            # bbs = box.bbox.tolist()
            # num = masks.size(0)
            # for i in range(num):
            #     bb1 = bbs[i]
            #     for j in range(i + 1, num):
            #         bb2 = bbs[j]
            #         if bb1[0] > bb2[2] or bb2[0] > bb1[2] or bb1[1] > bb2[3] or bb2[1] > bb1[3]:
            #             continue
            #         x_min = math.floor(min(bb1[0], bb2[0]))
            #         x_max = math.ceil(max(bb1[2], bb2[2]))
            #         y_min = math.floor(min(bb1[1], bb2[1]))
            #         y_max = math.ceil(max(bb1[3], bb2[3]))
            #
            #         overlap = masks[i][0][y_min:y_max, x_min:x_max] & masks[j][0][y_min:y_max, x_min:x_max]
            #         is_overlap = bool((overlap > 0).any())
            #         if is_overlap:
            #             if rel_val[i] < rel_val[j]:
            #                 masks[j][0][y_min:y_max, x_min:x_max] = masks[j][0][y_min:y_max, x_min:x_max] - overlap
            #             elif rel_val[j] < rel_val[i]:
            #                 masks[i][0][y_min:y_max, x_min:x_max] = masks[i][0][y_min:y_max, x_min:x_max] - overlap
            #
            # box.add_field("mask", masks)

        # print(time.time()-t1)
        return boxes

def make_roi_relation_post_processor(cfg):
    relation_post_processor = RelationPostProcessor()
    return relation_post_processor
