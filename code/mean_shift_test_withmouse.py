import sys
import numpy as np
import cv2

path = r'' # source path
path_1 = r'' # test image path
path_2 = r'' # test image path
path_3 = r'' # test image path
path_4 = r'' # test image path
path_5 = r'' # test image path

path_list = [path_1, path_2, path_3, path_4, path_5]

src = cv2.imread(path)

x, y, w, h = cv2.selectROI(src)

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
crop = src_ycrcb[y:y+h, x:x+w]

channels = [1, 2]
cr_bins = 128
cb_bins = 128
histsize = [cr_bins, cb_bins]
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range

hist = cv2.calcHist([crop], channels, None, histsize, ranges)
hist_norm = cv2.normalize(cv2.log(hist + 1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

for idx, path_ in enumerate(path_list):
    img_read = cv2.imread(path_)
    src_ycrcb_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2YCrCb)

    hist_norm = cv2.normalize(cv2.log(hist + 1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    backproj = cv2.calcBackProject([src_ycrcb_read], channels, hist, ranges, 1)

    dst = cv2.copyTo(img_read, backproj)

    cv2.imshow('backproj', backproj)
    cv2.imshow('hist_norm', hist_norm)
    cv2.imshow('dst', dst)
    cv2.imwrite(rf'', dst)
    cv2.waitKey()

    