import sys
import os
import cv2
import numpy as np

path = r'' # image path
src = cv2.imread(path)
rc = cv2.selectROI(src)
mask = np.zeros(src.shape[:2], np.uint8)

cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)
# cv2.GC_BGD: 확실한 배경(0), cv2.GC_FGD: 확실한 전경(1), cv2.GC_PR_BGD: 아마도 배경(2), cv2.GC_PR_FGD: 아마도 전경(3)
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask2[:,:,np.newaxis]

cv2.imshow('dst', dst)
cv2.waitKey()