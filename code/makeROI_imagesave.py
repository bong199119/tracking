import sys
import os
import cv2
import numpy as np

imgname = 'test4'

path = rf''
src = cv2.imread(path)
x, y, w, h = cv2.selectROI(src)

print(x, y, w, h)
cropped_image = src[y:y+h, x:x+w]

cv2.imwrite(rf'', cropped_image)
cv2.imshow('dst', cropped_image)
cv2.waitKey(0)