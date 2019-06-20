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
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from pycocotools import mask



def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

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
        # T.ToPILImage(),
        # T.Resize(min_image_size),
        # T.ToTensor(),
        to_bgr_transform,
        normalize_transform,
    ])
    return transform

# config
config_file = "configs/e2e_mask_rcnn_R_50_FPN_1x_relation_finetune_Tencent.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "OUTPUT_DIR", "output_relation_finetune/","TEST.IMS_PER_BATCH","1"])

# .uftmp

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
    0.7,0,
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

    try:
        with torch.no_grad():
            outputs = model(images)
    except:
        continue
    image_list = images.tensors.to(cpu_device).numpy()
    # outputs = [o.to(cpu_device) for o in outputs]
    targets = [target.to(device) for target in targets]
    # reshape prediction (a BoxList) into the original image size


    masker = Masker(threshold=mask_threshold, padding=1)

    for image, output, target in zip(image_list, outputs, targets):
        scores = output.get_field("scores")
        keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
        output = output[keep]
        # scores = output.get_field("scores")
        # _, idx = scores.sort(0, descending=True)
        # output = output[idx]

        image = image.transpose(1,2,0)
        image = np.resize(image, (600, 960, 3))
        image += np.array(cfg.INPUT.PIXEL_MEAN)

        overlay_image = image.copy()
        output_image = image.copy()

        try:
            match_quality_matrix = boxlist_iou(target, output)
            matched_idxs = matcher(match_quality_matrix).tolist()
        except:
            continue

        if output.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = output.get_field("mask")
            # always single image is passed at a time
            masks = masker([masks], [output])[0]
            output.add_field("mask", masks)

        pred_vals = output.get_field("relation_val").tolist()
        target_instance_ids = target.get_field("instance_ids").tolist()
        pred_instance_ids = []
        for idx in matched_idxs:
            if idx > 0:
                pred_instance_ids.append(target_instance_ids[idx])
            else:
                pred_instance_ids.append(-1)
        target_relations = target.get_field("relations")["relations"]

        masks = output.get_field("mask").numpy()
        num = len(masks)
        print(len(pred_instance_ids), num)
        flag = False
        for i in range(num):
            instance_id1 = pred_instance_ids[i]
            for j in range(i+1, num):
                instance_id2 = pred_instance_ids[j]
                # overlap
                overlap = cv2.bitwise_and(masks[i][0], masks[j][0])
                overlap_encoded = mask.encode(np.asfortranarray(overlap))
                area = mask.area(overlap_encoded)
                if area:
                    if (instance_id1, instance_id2) in target_relations:
                        if pred_vals[i] < pred_vals[j]:
                            print("correct relation with overlap:   area: ", area, "  instance id is ", instance_id1, instance_id2)
                            # pass
                        else:
                            print("wrong relation with overlap:   area: ", area, "  instance id is ", instance_id1,
                                  instance_id2)
                            # if area > 0:
                            #     mask1 = masks[i][0]
                            #     mask2 = masks[j][0]
                            #
                            #     contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            #     overlay = cv2.fillPoly(overlay_image, contours, (50, 200, 0))
                            #
                            #     contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            #     overlay = cv2.fillPoly(overlay_image, contours, (50, 20, 200))
                            #
                            #     flag = True
                    elif (instance_id2, instance_id1) in target_relations:
                        if pred_vals[j] < pred_vals[i]:
                            print("correct relation with overlap:   area: ", area, "  instance id is ", instance_id1, instance_id2)
                            # pass
                        else:
                            print("wrong relation with overlap:   area: ", area, "  instance id is ", instance_id1,
                                  instance_id2)
                    if area > 0:
                        mask1 = masks[i][0]
                        mask2 = masks[j][0]

                        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        overlay = cv2.fillPoly(overlay_image, contours, (50, 200, 0))

                        contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        overlay = cv2.fillPoly(overlay_image, contours, (50, 20, 200))

                        flag = True
                    # else:
                        # print("unknown relation with overlap:   area: ", area, "  instance id is ", instance_id1, instance_id2)
                        # if area > 20:
                        #     mask1 = masks[i][0]
                        #     mask2 = masks[j][0]
                        #
                        #     contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #     overlay = cv2.fillPoly(overlay_image, contours, (50, 200, 0))
                        #
                        #     contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        #     overlay = cv2.fillPoly(overlay_image, contours, (50, 20, 200))
                        #
                        #     flag = True

                        # break
            # if flag:
            #     break
        if not flag:
            continue
        alpha = 0.5
        cv2.addWeighted(overlay_image, alpha, output_image, 1 - alpha, 0, output_image)
        cv2.imwrite("img.jpg", output_image)
    # if ni == 2:
    #     t2 = time.time()
    #     print(t2-t1, ni)
    #     t1 = time.time()
    break

# print(all_acc_num/all_count)
