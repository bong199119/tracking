import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import time
import cv2

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def hex2RGB(color_hex):
    color_hex = color_hex.lstrip('#')
    color_RGB = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    return color_RGB

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    alpha = 0.8 # 높을수록 라벨링 투명도가 높음

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name) # 원본
    image_origin = io.imread(image_name) # 원본
    image_for_forg = io.imread(image_name)
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
        img_read = cv2.fillPoly(image, [list_polygons], color_RGB)
    dst = cv2.addWeighted(image_origin, alpha, img_read, (1-alpha), 0) 
    # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    ###############################################

    dst_foreground = cv2.copyTo(image_for_forg, gray_mask_array)
    foreground_img = Image.fromarray(dst_foreground)
    fillpoly_img = Image.fromarray(dst)
    path_to_save_file = os.path.join(d_dir, imidx+'_mask.png')
    path_to_save_file_fillpoly_img = os.path.join(d_dir, imidx+'_inference.png')
    path_to_save_file_foreground_img = os.path.join(d_dir, imidx+'_foreground.png')

    imo.save(path_to_save_file)
    foreground_img.save(path_to_save_file_foreground_img)
    fillpoly_img.save(path_to_save_file_fillpoly_img)

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    dict_models = { 
                    '2.0_250000' : r'/workspace/U-2-Net/saved_models/training_data_2.0/u2net_bce_itr_250000_train_0.184458_tar_0.019536.pth',
                    }

    # prediction_root_dir = r'/workspace/U-2-Net/results/model1.3_testdata3.0.1'

    for model_num_epoch in dict_models:
        model_num = model_num_epoch.split('_')[-1]
        model_version = model_num_epoch.split('_')[0]
        print('model_num : ', model_num)
        model_dir = dict_models[model_num_epoch]

        path_videos = r'/workspace/U-2-Net/videos/for_trackingsmaple'
        video_list = [file_ for file_ in os.listdir(path_videos) if file_.endswith('mp4')]

        for video_ in video_list:
            video_output_folder = os.path.join(path_videos, video_[:-4])
            os.makedirs(video_output_folder, exist_ok=True)

            video_path = os.path.join(path_videos, video_)
            cap = cv2.VideoCapture(video_path)

            vid_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    vid_idx_08d = (8-len(str(vid_idx)))*'0'+str(vid_idx)
                    path_image_forinference = os.path.join(video_output_folder,video_[:-4]+f'_{vid_idx_08d}.png')
                    cv2.imwrite(path_image_forinference, frame)
                    print('save ', path_image_forinference)
                    vid_idx += 1
                else:
                    break

            img_name_list = [png_ for png_ in glob.glob(video_output_folder + os.sep + '*') if png_.endswith('png')] # or jpg
            print(img_name_list)

            # --------- 2. dataloader ---------
            #1. dataloader
            test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                                lbl_name_list = [],
                                                transform=transforms.Compose([RescaleT(320),
                                                                            ToTensorLab(flag=0)])
                                                )
            test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1)

            # --------- 3. model define ---------
            if(model_name=='u2net'):
                print("...load U2NET---173.6 MB")
                net = U2NET(3,1)
            elif(model_name=='u2netp'):
                print("...load U2NEP---4.7 MB")
                net = U2NETP(3,1)

            if torch.cuda.is_available():
                net.load_state_dict(torch.load(model_dir))
                net.cuda()
            else:
                net.load_state_dict(torch.load(model_dir, map_location='cpu'))
            net.eval()

            # --------- 4. inference for each image ---------

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

                save_output(img_name_list[i_test], pred, video_output_folder)
                del d1,d2,d3,d4,d5,d6,d7
        
            list_image_forvideo = [image_ for image_ in os.listdir(video_output_folder) if image_[-13:] == 'inference.png']
            list_image_forvideo = sorted(list_image_forvideo)

            list_image_forvideo_forg = [image_ for image_ in os.listdir(video_output_folder) if image_[-14:] == 'foreground.png']
            list_image_forvideo_forg = sorted(list_image_forvideo_forg)

            print(list_image_forvideo)
            ###
            fps = 30
            image_sample = io.imread(os.path.join(video_output_folder, img_name_list[0]))
            vid_w, vid_h = image_sample.shape[1], image_sample.shape[0]
            fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
            video_output_path = os.path.join(path_videos, f'{video_[:-4]}_inferenced.mp4')
            video_output_path_forg = os.path.join(path_videos, f'{video_[:-4]}_foreground.mp4')
            vid_out = cv2.VideoWriter(video_output_path, fourcc, fps, (vid_w, vid_h))
            vid_out_forg = cv2.VideoWriter(video_output_path_forg, fourcc, fps, (vid_w, vid_h))
            ###
            for image_ in list_image_forvideo:
                path_image = os.path.join(video_output_folder, image_)
                print(image_)
                imread = cv2.imread(path_image)
                vid_out.write(imread)
            
            for image_ in list_image_forvideo_forg:
                path_image = os.path.join(video_output_folder, image_)
                print(image_)
                imread = cv2.imread(path_image)
                vid_out_forg.write(imread)

            vid_out.release()
            vid_out_forg.release()

if __name__ == "__main__":
    main()