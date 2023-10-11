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
# all_files   = os.listdir(path_vidieos_folder)
# list_videos = [ fname for fname in all_files if fname.endswith('.mp4')]
list_videos = [
]

# dict_tracker = {
#     'csrt' : cv2.legacy.TrackerCSRT_create(),
#     'kcf' : cv2.legacy.TrackerKCF_create(), 
#     'boosting' : cv2.legacy.TrackerBoosting_create(),
#     'mil' : cv2.legacy.TrackerMIL_create(),
#     'tld' : cv2.legacy.TrackerTLD_create(),
#     'medianflow' : cv2.legacy.TrackerMedianFlow_create(),
#     'mosse' : cv2.legacy.TrackerMOSSE_create()
# }

print('[ CV ver.', cv2.__version__, ']')
for videos_ in list_videos:
    print(list_videos)
    #video_path = os.path.join(video_folder_path, videos_)
    # video_path = './input/output_detecting_5_src.mp4';
    video_path = os.path.join(path_vidieos_folder, videos_)
    #csv_path = os.path.join(video_folder_path, videos_[:-3]+'csv')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_path)
    time.sleep(5)
    
    # Loop through the video frames
    vid_w   = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
    vid_h   = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(' > SOURCE: ', video_path)
    print(' > WIDTH: ',  vid_w);
    print(' > HEIGHT: ', vid_h);
    print(' > FPS: ',    vid_fps);
    print(' > TOTAL FRAMES: ', length);
    
    frame_num = 0
    video_num = 1
    list_total = []
    # print('length',length)
    time.sleep(1)

    video_output_path = os.path.join(path_vidieos_folder, videos_[:-4]+f'_tracked_GOTURN'+videos_[-4:])
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(video_output_path, fourcc, fps, (vid_w, vid_h))
    #print(' > OUTPUT: ', video_output_path)
    """
    if not vid_out.isOpened():
        print(' > FAILED TO CREATE OUTPUT VIDEO')
        cap.release()
        sys.exit()
    """

    tracker_init = 0;
    
    # tracker = cv2.legacy.TrackerCSRT_create()
    tracker = cv2.TrackerGOTURN_create()
    # tracker = cv2.legacy.TrackerMOSSE_create()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if frame_num >= 0:
                if frame_num % 100 == 0:
                    print(' > frame_num: ',frame_num,', ', (frame_num*100) / length,'%')
                
                #list_temp = [frame_num,0,0,0,0,0,0,0,0,0,0,0,0]
                
                # Run YOLOv8 inference on the frame
                results = model(frame, verbose=False)
            
                det_frame = results[0].plot()
                #cv2.imwrite(path_results + f"/det_{frame_num}.png", det_frame)
                results = model(frame, verbose=False)
                #vid_out.write(frame)

                #if frame_num > 10:
                #    break;
                tracker_init = 0

                for result in results:
                    j = 0;
                    for box in result.boxes:
                        i = 0;
                        for cls in box.cls:
                            print(' > frame.',frame_num,', deteteced.', j);
                            obj_xy = box.xyxy[i].tolist()
                            cls_id = box.cls[i].item()
                            prob   = box.conf[i].item()
                            print('   - xyxy = ', [round(x) for x in obj_xy])
                            print('   - clsid= ', round(cls_id), ':', result.names[cls_id])
                            print('   - prob = ', round(prob), 2)
                            
                            # if tracker_init==0 and round(cls_id)==6:
                            #   tracker_init = 1
                            if round(cls_id)==2:
                            # if True:
                                if prob > 0.7:
                                    # targetimg = cv2.selectROI(result.names[cls_id], frame)
                                    #print(obj_xy[0],obj_xy[1],obj_xy[2],obj_xy[3])
                                    #targetimg = frame[
                                    #        round(obj_xy[1]):round(obj_xy[3]),
                                    #        round(obj_xy[0]):round(obj_xy[2])]
                                    targetbox = [
                                            round(obj_xy[0]),round(obj_xy[1]),
                                            round(obj_xy[2]-obj_xy[0]),round(obj_xy[3]-obj_xy[1])]
                                    #cv2.imwrite(path_results + f"/object_{frame_num}_{round(cls_id)}.png", targetimg)
                                    #####################
                                    #####################
                                    # tracker = cv2.legacy.TrackerCSRT_create()
                                    tracker = cv2.TrackerGOTURN_create()
                                    tracker.init(frame, targetbox)
                                    tracker_init = 1
                                    #####################
                                    #####################
                            i+=1;
                        j+=1;

                if frame_num > 0:
                    ret, rc = tracker.update(frame)
                    rc = tuple([int(_) for _ in rc])
                    #cv2.rectangle(frame, rc, (0, 0, 255), 2)
                    #cv2.imwrite(path_results + f"/track_{frame_num}.png", frame)
                    #####################
                    #####################
                    cv2.rectangle(det_frame, rc, (0, 0, 255), 2)
                    #####################
                    #####################
                    #cv2.imwrite(path_results + f"/track_{frame_num}.png", det_frame)
                    print('   - track = ', rc);
                    # if tracker_init == 1:
                    #         cv2.imwrite(f'./img/{frame_num}.jpg', frame)

                vid_out.write(det_frame)
                """ 
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    #masks = result.masks  # Masks object for segmentation masks outputs
                    #keypoints = result.keypoints  # Keypoints object for pose outputs
                    # probs = result.probs  # Class probabilities for classification outputs
                    # print('boxes', boxes)
                    # print('boxes.cls', boxes.cls)
                    # print(type(boxes.cls.detach().cpu().numpy()))
                    #if list(boxes.cls.detach().cpu().numpy()) != []:
                        # print('test', int(boxes.cls.detach().cpu().numpy()[0]))
                        # list_temp[int(boxes.cls.detach().cpu().numpy()[0])+1] = 1
                list_total.append(list_temp)
                """
            frame_num+=1
        else:
            break

    cap.release()
    vid_out.release()
    print(' > FINISHED');

