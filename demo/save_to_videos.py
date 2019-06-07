import cv2
import numpy as np
import glob

images = glob.glob('/home/jim/Documents/4ring_ms_result/*.jpg')
images = sorted(images)
height, width, layers = cv2.imread(images[0]).shape
size = (width, height)

out = cv2.VideoWriter('/home/jim/Documents/4ring_ms_demo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
# out = cv2.VideoWriter('/home/jim/Documents/4ring_demo.avi', 0, 25, size)

for filename in images:
    img = cv2.imread(filename)
    out.write(img)
    # print(filename)
out.release()
