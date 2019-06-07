import json
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import cv2
import tqdm
import time
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


CATEGORIES = [
    "__background",
    "car",
    "pedestrain",
    "rider",
    "traffic light",
    "pole",
    "traffic sign"
]

def build_transform(cfg):
    global min_image_size
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(min_image_size),
        T.ToTensor(),
        to_bgr_transform,
        normalize_transform,
    ])
    return transform

# config
config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x_relation_Tencent.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "OUTPUT_DIR", "../output_relation/"])

min_image_size = 600
confidence_threshold = 0.7
masks_per_dim = 2
mask_threshold = 0.5
cpu_device = torch.device("cpu")

# model
model = build_detection_model(cfg)
model.eval()
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

# other preparation
save_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)

transforms = build_transform(cfg)

matcher = Matcher(
    # cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
    # cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
    0,0,
    allow_low_quality_matches=False,
)

# main

instance_ids = []

data_loaders = make_data_loader(cfg, is_train=False, is_distributed=False)
data_loader = data_loaders[0]

all_acc_num = 0
all_count = 0
print(len(data_loader))
t1 = time.time()
for ni, (images, targets, image_ids) in enumerate(data_loader):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    targets = [target.to(device) for target in targets]

    for output, target in zip(outputs, targets):
        scores = output.get_field("scores")
        keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
        output = output[keep]
        try:
            match_quality_matrix = boxlist_iou(target, output)
            matched_idxs = matcher(match_quality_matrix).tolist()
        except:
            continue

        pred_vals = output.get_field("relation_val").tolist()
        target_isntance_ids = target.get_field("instance_ids").tolist()
        pred_instance_ids = []
        for idx in matched_idxs:
            if idx > 0:
                pred_instance_ids.append(target_isntance_ids[idx])
            else:
                pred_instance_ids.append(-1)

        target_relations = target.get_field("relations")["relations"]

        # evaluate
        acc_num = 0
        count = 0
        for i in range(len(pred_instance_ids)):
            instance_id1 = pred_instance_ids[i]
            for j in range(i+1, len(pred_instance_ids)):
                instance_id2 = pred_instance_ids[j]
                if (instance_id1, instance_id2) in target_relations:
                    if pred_vals[i] < pred_vals[j]:
                        acc_num += 1
                    count += 1
                elif (instance_id2, instance_id1) in target_relations:
                    if pred_vals[j] < pred_vals[i]:
                        acc_num += 1
                    count += 1
        # acc = acc_num/count
        all_acc_num += acc_num
        all_count += count
    if ni % 10 == 0:
        t2 = time.time()
        print(t2-t1, ni)
        t1 = time.time()

print(all_acc_num/all_count)
