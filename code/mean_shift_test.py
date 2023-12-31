import sys
import numpy as np
import cv2


path_video = r'' # video path
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()


# 초기 사각형 영역
x, y, w, h = 135, 220, 100, 100
rc = (x, y, w, h)

ret, frame = cap.read()

if not ret:
    sys.exit()


roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    _, rc = cv2.meanShift(backproj, rc, term_crit)
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('backproj', backproj)

    if cv2.waitKey(60) == 27:
        break

cap.release()
cv2.destroyAllWindows()