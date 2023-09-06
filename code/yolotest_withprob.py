# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sys
# import moviepy.editor as mp
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from ultralytics import YOLO
import cv2
import time

import os

start_time = time.time()

model = YOLO('/home/jkim/workspace/object_detection/yolov8/models/best_4thdataset.pt')  # pretrained YOLOv8n model
video_folder_path = r'/home/jkim/workspace/object_detection/yolov8/videos/for_trackingsample'
# path_results = r'/home/bseo/workspace/object_detection/yolov8/img'
# video_path = "/home/bseo/workspace/object_detection/yolov8/videos/clip.mp4"

list_videos = os.listdir(video_folder_path)

for videos_ in list_videos:
    video_path = os.path.join(video_folder_path, videos_)
    csv_path = os.path.join(video_folder_path, videos_[:-4]+'_with_prob'+'.csv')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through the video frames

    print(video_path)
    frame_num = 0
    list_total = []
    # print('length',length)
    time.sleep(1)
    while cap.isOpened():
        # print(frame_num)
        success, frame = cap.read()
        if success:
            if frame_num % 30 == 0:
                # Read a frame from the video
                if frame_num % 990 == 0:
                    print('frame_num', frame_num)
                list_temp = [frame_num,0,0,0,0,0,0,0,0,0,0,0,0]
                # Run YOLOv8 inference on the frame
                results = model(frame)
                
                # annotated_frame = results[0].plot()
                # cv2.imwrite(os.path.join(path_results, f"{frame_num}.png"), annotated_frame)
                # print(results)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        for idx, cls in enumerate(box.cls):
                            obj_xy = box.xyxy[idx].tolist()
                            cls_id = box.cls[idx].item()
                            prob   = box.conf[idx].item()
                            
                            if prob > 0.7:
                                list_temp[int(cls_id)+1] = 1

                list_total.append(list_temp)
            frame_num+=1
        else:
        # Break the loop if the end of the video is reached
            break

    df_total = pd.DataFrame(data = list_total, columns = ["Frame","Obturator (OBTL-UCUK)", "Trocar (TRCL-UCUK)", "Ligasure (COVIDIEN) (LGSL-MACV)", "Graspers", "Polymer Clip Applier (CLAL-UCUK)", "Dissector", "Gauze (GAUL-UCUK)", "Linear Stapler", "Needle Holder (AESCULAP) (NDHL-UCAE)", "Specimen Retrieval Bag (RTBL-UCUK)", "Suction Irrigator", "Drain (JPCL-UCUK)"])
    # df_total = pd.DataFrame(data = list_total, columns = ["Bleeding"])
    df_total.to_csv(csv_path, index = False)
    
end_time = time.time()

print('time for inference : ',end_time-start_time)