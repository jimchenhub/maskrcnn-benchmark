import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
import os
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor_tencent import COCODemo


config_file = "configs/e2e_mask_rcnn_R_50_FPN_1x_Tencent.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "OUTPUT_DIR", "output/"])

coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.7,
)

import json
with open("../../Tencent_segmentation_annotations/instances_val2019.json", "r") as f:
    content = json.load(f)
images = content["images"]
names = [img["file_name"] for img in images][::50]


# ----------
# 4ring
# folder = "/home/jim/Documents/data_4ring/20180312_56_01/6mm_allframe/"
# images = os.listdir(folder)
# images = sorted(images)
# names = [os.path.join(folder, img) for img in images]
# print(len(names))
# ----------

count = 0
import time
t1 = time.time()
for name in names:
    output_name = name.split("/")[-1]
    # if os.path.isfile("/home/jim/Documents/4ring_mask_relation_finetune_result/"+output_name):
    #     continue
    pil_image = Image.open("../../Tencent_segmentation/" + name)
    # pil_image = Image.open(name)
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    # compute predictions
    predictions = coco_demo.run_on_opencv_image(image)
    cv2.imwrite("../../mask_vis/"+output_name, predictions[:,:,])
    # cv2.imwrite("/home/jim/Documents/4ring_mask_relation_finetune_result/"+output_name, predictions[:,:,])
    print(output_name)
    # break
    count += 1
    if count % 10 == 0:
        t2 = time.time()
        print(t2-t1, count)
        t1 = time.time()
    # if count == 1000:
    #     break
