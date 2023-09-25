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

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name) # 원본
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

    channels = [0, 1]
    roi_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [0, 180, 0, 256]
    hist = cv2.calcHist([roi_hsv], channels, gray_mask_array, [90, 128], ranges)
    print(hist)

    backproj = cv2.calcBackProject([image], channels, hist, ranges, 1)
    dst = cv2.copyTo(image, backproj)
    backproj_img = Image.fromarray(dst)

    path_to_save_file = os.path.join(d_dir, imidx+'.png')
    path_to_save_file_bpj = os.path.join(d_dir, imidx+'_.png')
    imo.save(path_to_save_file)
    backproj_img.save(path_to_save_file_bpj)

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    dict_models = { 
                    '{version}_{epoch}' : r'{model_path}',
                    }

    # prediction_root_dir = r'/workspace/U-2-Net/results/model1.3_testdata3.0.1'

    for model_num_epoch in dict_models:
        model_num = model_num_epoch.split('_')[-1]
        model_version = model_num_epoch.split('_')[0]
        dataset_name = 'test_data_fortracking_1.1'

        prediction_root_dir = rf'/workspace/U-2-Net/results/model{model_version}_{dataset_name}'

        print('model_num : ', model_num)
        model_dir = dict_models[model_num_epoch]
        image_dir = rf'/workspace/U-2-Net/data/{dataset_name}'
        prediction_dir = os.path.join(prediction_root_dir, model_num)
        os.makedirs(prediction_dir, exist_ok=True)

        img_name_list = [png_ for png_ in glob.glob(image_dir + os.sep + '*') if png_.endswith('png')] # or jpg
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

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)
            del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()