# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sys
# import moviepy.editor as mp
from tqdm import tqdm
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from ultralytics import YOLO
import cv2
import os

model = YOLO('')  # pretrained YOLOv8n model

path_vidieos_folder = r''
list_videos = [
    '',
    '',
    '',
]

dict_tracker = {
    'csrt' : cv2.legacy.TrackerCSRT_create(),
    'kcf' : cv2.legacy.TrackerKCF_create(), 
    'boosting' : cv2.legacy.TrackerBoosting_create(),
    'mil' : cv2.legacy.TrackerMIL_create(),
    'tld' : cv2.legacy.TrackerTLD_create(),
    'medianflow' : cv2.legacy.TrackerMedianFlow_create(),
    'mosse' : cv2.legacy.TrackerMOSSE_create()
}

print('[ CV ver.', cv2.__version__, ']')
for tracker_ in dict_tracker:
    for videos_ in list_videos:
        print(list_videos)
        video_path = os.path.join(path_vidieos_folder, videos_)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(video_path)
        time.sleep(5)
        
        vid_w   = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h   = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        print(' > SOURCE: ', video_path)
        print(' > WIDTH: ',  vid_w)
        print(' > HEIGHT: ', vid_h)
        print(' > FPS: ',    vid_fps)
        print(' > TOTAL FRAMES: ', length)
        
        frame_num = 0
        video_num = 1
        list_total = []
        time.sleep(1)

        video_output_path = os.path.join(path_vidieos_folder, videos_[:-4]+f'_tracked with {tracker_}_'+videos_[-4:])
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter(video_output_path, fourcc, fps, (vid_w, vid_h))

        tracker_init = 0
        tracker = dict_tracker[tracker_]

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if frame_num >= 0:
                    if frame_num % 100 == 0:
                        print(' > frame_num: ',frame_num,', ', (frame_num*100) / length,'%')
                    
                    results = model(frame, verbose=False)
                
                    det_frame = results[0].plot()
                    results = model(frame, verbose=False)
                    tracker_init = 0

                    for result in results:
                        j = 0;
                        for box in result.boxes:
                            i = 0;
                            for cls in box.cls:
                                print(' > frame.',frame_num,', deteteced.', j)
                                obj_xy = box.xyxy[i].tolist()
                                cls_id = box.cls[i].item()
                                prob   = box.conf[i].item()
                                print('   - xyxy = ', [round(x) for x in obj_xy])
                                print('   - clsid= ', round(cls_id), ':', result.names[cls_id])
                                print('   - prob = ', round(prob), 2)
                                
                                if round(cls_id)==6:
                                    if prob > 0.7:
                                        targetbox = [
                                                round(obj_xy[0]),round(obj_xy[1]),
                                                round(obj_xy[2]-obj_xy[0]),round(obj_xy[3]-obj_xy[1])]
                                        tracker = dict_tracker[tracker_]
                                        tracker.init(frame, targetbox)
                                        tracker_init = 1
                                i+=1
                            j+=1

                    if frame_num > 0:
                        ret, rc = tracker.update(frame)
                        rc = tuple([int(_) for _ in rc])
                        cv2.rectangle(det_frame, rc, (0, 0, 255), 2)
                        print('   - track = ', rc)

                    vid_out.write(det_frame)
                frame_num+=1
            else:
                break

        cap.release()
        vid_out.release()
        print(' > FINISHED')

