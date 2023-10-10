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
from torch.autograd import Variable
# import moviepy.editor as mp
from tqdm import tqdm
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from ultralytics import YOLO
import cv2
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from skimage import io, transform
from deep_sort_realtime.deepsort_tracker import DeepSort


"""
deep sort realtime Parameters 
pypi link : https://pypi.org/project/deep-sort-realtime/
----------
max_iou_distance : Optional[float] = 0.7
    Gating threshold on IoU. Associations with cost larger than this value are
    disregarded. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
max_age : Optional[int] = 30
    Maximum number of missed misses before a track is deleted. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
n_init : int
    Number of frames that a track remains in initialization phase. Defaults to 3. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
nms_max_overlap : Optional[float] = 1.0
    Non-maxima suppression threshold: Maximum detection overlap, if is 1.0, nms will be disabled
max_cosine_distance : Optional[float] = 0.2
    Gating threshold for cosine distance
nn_budget :  Optional[int] = None
    Maximum size of the appearance descriptors, if None, no budget is enforced
gating_only_position : Optional[bool]
    Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
override_track_class : Optional[object] = None
    Giving this will override default Track class, this must inherit Track. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
embedder : Optional[str] = 'mobilenet'
    Whether to use in-built embedder or not. If None, then embeddings must be given during update.
    Choice of ['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101', 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
half : Optional[bool] = True
    Whether to use half precision for deep embedder (applicable for mobilenet only)
bgr : Optional[bool] = True
    Whether frame given to embedder is expected to be BGR or not (RGB)
embedder_gpu: Optional[bool] = True
    Whether embedder uses gpu or not
embedder_model_name: Optional[str] = None
    Only used when embedder=='torchreid'. This provides which model to use within torchreid library. Check out torchreid's model zoo.
embedder_wts: Optional[str] = None
    Optional specification of path to embedder's model weights. Will default to looking for weights in `deep_sort_realtime/embedder/weights`. If deep_sort_realtime is installed as a package and CLIP models is used as embedder, best to provide path.
polygon: Optional[bool] = False
    Whether detections are polygons (e.g. oriented bounding boxes)
today: Optional[datetime.date]
    Provide today's date, for naming of tracks. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
"""
object_tracker = DeepSort(
    max_age = 15,
    # nms_max_overlap = 1.0,
    n_init = 3, 
    max_cosine_distance = 0.3, 
    # nn_budget = None,
    # override_track_class = None,
    # embedder = 'mobilenet'
    # half = True,
    # bgr = True,
    # embedder_gpu = True,
    # embedder_model_name = True,
    # embedder_wts = None,
    # polygone = False,
    # today = None,
)

model = YOLO('')  # pretrained YOLOv8n model

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def hex2RGB(color_hex):
    color_hex = color_hex.lstrip('#')
    color_RGB = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    return color_RGB

def return_output(image_name,pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    alpha = 0.8 # 높을수록 라벨링 투명도가 높음

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name) # 원본
    image_origin = io.imread(image_name) # 원본
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR) # mask(pred)
    pb_np = np.array(imo)
    
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    # mask = np.zeros(image.shape[:2],np.uint8)
    # mask[100:150,100:200] = 255

    gray_mask  = imo.convert('L')
    gray_mask_array = np.array(gray_mask)

    # find contours from mask(array)##############
    contours, _ = cv2.findContours(gray_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for obj in contours:
        polygons = []
        polygons_tmp = []
        
        for point in obj:
            coords = []
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

            polygons_tmp.append(coords)
        polygons.append(polygons_tmp)

        list_polygons = np.array(polygons, dtype=np.int32)
        color_hex = '#388E3C'
        color_RGB = hex2RGB(color_hex)
        # img_read = cv2.fillPoly(image, [list_polygons], color_RGB)
    # dst = cv2.addWeighted(image_origin, alpha, img_read, (1-alpha), 0) 
    # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    ###############################################
    dst = cv2.copyTo(image, gray_mask_array)
    image_name_foreground = image_name[:-4] + '_foreground' + image_name[-4:]
    cv2.imwrite(image_name_foreground, dst)
    # foreground_img = Image.fromarray(dst)
    # path_to_save_file = os.path.join(d_dir, imidx+'_mask.png')
    # path_to_save_file_foreground = os.path.join(d_dir, imidx+'_inference.png')
    # imo.save(path_to_save_file)
    # foreground_img.save(path_to_save_file_foreground)

    return dst

path_model_u2net = r''
path_vidieos_folder = r''
# all_files   = os.listdir(path_vidieos_folder)
# list_videos = [ fname for fname in all_files if fname.endswith('.mp4')]
list_videos = [
    ''
    ]

model_name='u2net'#u2netp
if(model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
elif(model_name=='u2netp'):
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(path_model_u2net))
    net.cuda()
else:
    net.load_state_dict(torch.load(path_model_u2net, map_location='cpu'))
net.eval()

print('[ CV ver.', cv2.__version__, ']')
for videos_ in list_videos:
    path_inferencedfolder = os.path.join(path_vidieos_folder, videos_[:-4])
    os.makedirs(path_inferencedfolder, exist_ok=True)

    print(list_videos)
    video_path = os.path.join(path_vidieos_folder, videos_)
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
    time.sleep(1)

    video_output_path = os.path.join(path_vidieos_folder, videos_[:-4]+f'_bitracked_deepsort_mg15_ninit3_cos0.3'+videos_[-4:])
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(video_output_path, fourcc, fps, (vid_w, vid_h))
    """
    if not vid_out.isOpened():
        print(' > FAILED TO CREATE OUTPUT VIDEO')
        cap.release()
        sys.exit()
    """

    tracker_init = 0;
    
    # tracker = cv2.legacy.TrackerCSRT_create()
    # tracker = cv2.TrackerGOTURN_create()
    tracker = object_tracker
    # tracker = cv2.legacy.TrackerMOSSE_create()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if frame_num >= 0:
                if frame_num % 100 == 0:
                    print(' > frame_num: ',frame_num,', ', (frame_num*100) / length,'%')
                
                results = model(frame, verbose=False)
                ###############################################################
                vid_idx_08d = (8-len(str(frame_num)))*'0'+str(frame_num)
                path_img_ele = os.path.join(path_inferencedfolder, videos_[:-4]+f'_{vid_idx_08d}.png')
                cv2.imwrite(path_img_ele, frame)
                img_name_list = [path_img_ele]

                test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                                lbl_name_list = [],
                                                transform=transforms.Compose([RescaleT(320),
                                                                            ToTensorLab(flag=0)])
                                                )
                test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=1)
                
                

                for i_test, data_test in enumerate(test_salobj_dataloader):

                    print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
                    inputs_test = data_test['image']
                    inputs_test = inputs_test.type(torch.FloatTensor)
                    if torch.cuda.is_available():
                        inputs_test = Variable(inputs_test.cuda())
                    else:
                        inputs_test = Variable(inputs_test)

                    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

                    # normalization
                    pred = d1[:,0,:,:]
                    pred = normPRED(pred)

                    frame_foreground = return_output(img_name_list[i_test], pred)
                    del d1,d2,d3,d4,d5,d6,d7
                ###############################################################

                det_frame = results[0].plot()
                tracker_init = 0

                list_for_deepsort = []
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
                            targetbox = [
                                            round(obj_xy[0]),round(obj_xy[1]),
                                            round(obj_xy[2]-obj_xy[0]),round(obj_xy[3]-obj_xy[1])]
                            
                            list_for_deepsort.append((targetbox, prob, cls_id))
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
                                    # tracker = cv2.TrackerGOTURN_create()
                                    # tracker.init(frame_foreground, targetbox)

                                    # object_tracker.update_tracks(targetbox, frame = frame_foreground)
                                    # tracker_init = 1
                                    #####################
                                    #####################
                                #####################
                                targetbox = [
                                            round(obj_xy[0]),round(obj_xy[1]),
                                            round(obj_xy[2]-obj_xy[0]),round(obj_xy[3]-obj_xy[1])]
                                
                                
                                #####################
                            i+=1;
                        j+=1;

                if frame_num > 0:
                    print('before track')
                    tracks = tracker.update_tracks(list_for_deepsort, frame = frame_foreground)
                    print('after track')
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        bbox = ltrb
                        cv2.rectangle(det_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    #####################
                    #####################
                    # ret, rc = tracker.update(frame_foreground)
                    # #cv2.rectangle(frame, rc, (0, 0, 255), 2)
                    # #cv2.imwrite(path_results + f"/track_{frame_num}.png", frame)
                    # cv2.rectangle(det_frame, rc, (0, 0, 255), 2)
                    #####################
                    #####################
                    #cv2.imwrite(path_results + f"/track_{frame_num}.png", det_frame)
                    
                    # print('   - track = ', rc);
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

