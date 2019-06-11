import cv2
import time
import numpy as np
import glob

images = glob.glob('/home/jim/Documents/4ring_mask_relation_finetune_result/*.jpg')
images = sorted(images)
height, width, layers = cv2.imread(images[0]).shape
size = (width//2, height//2)

out = cv2.VideoWriter('/home/jim/Documents/4ring_mask_relation_finetune_result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
# out = cv2.VideoWriter('/home/jim/Documents/4ring_demo.avi', 0, 25, size)

print(len(images))
count = 0
t1 = time.time()
for filename in images:
    img = cv2.imread(filename)
    img = cv2.resize(img, size)
    out.write(img)
    count += 1
    if count % 100 == 0:
        t2 = time.time()
        print(t2-t1, count)
        t1 = time.time()
    # print(filename)
out.release()
