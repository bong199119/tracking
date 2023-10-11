import os
import pandas as pd
import numpy as np
import time

def frame_to_hms(idx, fram_gap, fps):
    sec = idx * (fram_gap/fps)
    sec = round(sec)

    h = int(sec//3600)
    sec = sec-h*3600
    m = int(sec//60)
    sec = sec-m*60

    if len(str(h)) == 1:
        h_str = '0'+str(h)
    else:
        h_str = str(h)
    if len(str(m)) == 1:
        m_str = '0'+str(m)
    else:
        m_str = str(m)
    # if len(str(sec)) != 4:
    #     s_str = '0'+str(sec)
    # else:
    #     s_str = str(sec)
    if len(str(sec)) == 1:
        s_str = '0'+str(sec)
    else:
        s_str = str(sec)

    hms = h_str +':'+m_str+':'+s_str
    return hms

def sec_to_hms(sec):
    h = int(sec//3600)
    sec = sec-h*3600
    m = int(sec//60)
    sec = sec-m*60

    if len(str(h)) == 1:
        h_str = '0'+str(h)
    else:
        h_str = str(h)
    if len(str(m)) == 1:
        m_str = '0'+str(m)
    else:
        m_str = str(m)

    if len(str(sec)) == 1:
        s_str = '0'+str(sec)
    else:
        s_str = str(sec)

    hms = h_str +':'+m_str+':'+s_str
    return hms

def hms_to_sec(hms):
    h = int(hms.split(':')[0])
    m = int(hms.split(':')[1])
    s = int(hms.split(':')[2])

    if h < 0:
        raise Exception('not available number for h')
    
    if (len(hms.split(':')[1]) != 2) or (m < 0) or (m > 60):
        raise Exception('not available number for m')
        
    if (len(hms.split(':')[2]) != 2) or (s < 0) or (s > 60):
        raise Exception('not available number for s')
    
    sec = (h * 3600) + (m * 60) + s
    return sec

def hms_to_hmsWC(hms):
    h = hms.split(':')[0]
    m = hms.split(':')[1]
    s = hms.split(':')[2]

    hmswc = h+m+s
    return hmswc

def packaging_hms_list(hms_list):
    sec_list = []
    for hms_ele in hms_list:
        sec = hms_to_sec(hms_ele)
        sec_list.append(sec)

    before_sec = ''
    list_stack = []
    hms_list_packaged = []
    for idx, sec_ele in enumerate(sec_list):
        if before_sec == '':
            before_sec = sec_ele
            list_stack.append(before_sec)

        else:
            if before_sec+1 == sec_ele:
                list_stack.append(sec_ele)
                
            if (before_sec+1 != sec_ele) or (idx == len(sec_list)-1):
                hms_list_packaged.append(f'{sec_to_hms(list_stack[0])} ~ {sec_to_hms(list_stack[-1])}')
                list_stack = [sec_ele]

            before_sec = sec_ele

    return hms_list_packaged

def hms_to_duetime(hmspck):
    first_hms = hmspck.split('~')[0][:-1]
    second_hms = hmspck.split('~')[1][1:]
    
    first_sec = hms_to_sec(first_hms)
    second_sec = hms_to_sec(second_hms)

    duetime = second_sec - first_sec
    
    return duetime  


gt_path = r''
video_path = r''
tool_name = ''
tool_name_forfolder = tool_name.replace(' ','_')
print(tool_name)
# time.sleep(100)
clip_output_path = rf''
os.makedirs(clip_output_path, exist_ok=True)
list_gtfile = [
    '',
    '',
    '',
    '',
]

fram_gap = 30
fps = 30

for gtfile in list_gtfile:
    path_gtfile = os.path.join(gt_path, gtfile)
    df_gt = pd.read_csv(path_gtfile)
    list_toolappear = df_gt[tool_name]

    list_toolappear_hmst = []
    for index, ele in enumerate(list_toolappear):
        if ele == 1:
            hms = frame_to_hms(index, fram_gap, fps)
            list_toolappear_hmst.append(hms)
            
    hms_list_packaged = packaging_hms_list(list_toolappear_hmst)

    for hms_packaged in hms_list_packaged:
        
        adjust_front = 3
        adjust_back = 3

        hms_start_time = hms_packaged.split(' ~ ')[0]
        hms_end_time = hms_packaged.split(' ~ ')[1]
        print('first', hms_start_time)
        print('first', hms_end_time)

        if hms_to_sec(hms_start_time) - adjust_front > 0:
            hms_start_time = sec_to_hms(hms_to_sec(hms_start_time) - adjust_front)
        elif hms_to_sec(hms_start_time) - adjust_front <= 0:
            hms_start_time = hms_start_time
        
        hms_end_time = sec_to_hms(hms_to_sec(hms_end_time) + adjust_back)
        hmswc_start_time = hms_to_hmsWC(hms_start_time)
        hmswc_end_time = hms_to_hmsWC(hms_end_time)

        print('second', hms_start_time)
        print('second', hms_end_time)
        try:
            path_video = os.path.join(video_path, gtfile[:24]+'.mp4')
            path_output_clip = os.path.join(clip_output_path, gtfile[:24]+f'_{hmswc_start_time}_{hmswc_end_time}.mp4')
            
            duetime = hms_to_sec(hms_end_time) - hms_to_sec(hms_start_time)
            command = f"ffmpeg -i {path_video} -ss {hms_start_time} -t {duetime} -vcodec libx264 -acodec copy {path_output_clip}"
            os.system(command)
            print(command)
        except:
            print('fail to make clip')
        
        
        
