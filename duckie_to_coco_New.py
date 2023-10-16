# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:54:36 2020

@author: yishu
"""

import numpy as np
import matplotlib.pyplot as plt 
import cv2

#img_num="5"
train = True

#dataset_name = "real"
#dataset_name = "sim"
dataset_name = "js"
if dataset_name=="real":
    image_folder="../data/dt_segmentation_dataset/real/imgs/"
    mask_folder="../data/dt_segmentation_dataset/real/labels/"
    ext="jpg"
    img_num="frame0"
    real=True
if dataset_name=="sim":
    image_folder="../data/dt_segmentation_dataset/sim/rgb_160_120/"
    mask_folder="../data/dt_segmentation_dataset/sim/label_160_120/"
    img_num="1"
    ext="jpg"
    real=False
if dataset_name=="js":
    image_folder = "C:/segmentation_code/Yishu/dataset_js_02/image_2020_12_04_11_25_04/w_bezier/rgb_orig/"
    mask_folder = "C:/segmentation_code/Yishu/dataset_js_02/image_2020_12_04_11_25_04/w_bezier/labels/"
    img_num="42"
    ext="png"
    real = False
print(real)

import random
r = lambda: random.randint(0,255)

def random_color():
    return (r(),r(),r())
#thresh
    
real2sim = {0:0, 1:3, 2:4, 3:5, 4:2, 5:1}
js2sim = {0:0, 1:2, 2:1, 3:3, 4:4, 5:5}
#image = cv2.imread(f'{image_folder}/{img_num}.jpg')[:,:,::-1]
print(f'{image_folder}/{img_num}.{ext}')

import os
print(os.path.exists(f'{image_folder}/{img_num}.{ext}'))

mask_flattened = np.load(f'{mask_folder}/{img_num}.npy')
if dataset_name == "js":
    mask_flattened = cv2.resize(mask_flattened.reshape(240,320).astype(np.uint8), (320,240), cv2.INTER_NEAREST)

image2 = 0
h, w = 320, 240
mask_reshaped = mask_flattened.reshape(w,h)
print("Mask shape: ",mask_reshaped.shape)
image = cv2.imread(f'{image_folder}/{img_num}.{ext}')
if dataset_name == "js":
    image = cv2.resize(image, (320,240))
    print("Image Shape: ",image.shape)
from cv2_plt_imshow import cv2_plt_imshow as cv2_imshow
for i in range(1,6):
    thresh = (mask_reshaped==i).astype(np.uint8)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #cnt = contours[1]
    if real:
        i = real2sim[i]
    if dataset_name=="js":
        i = js2sim[i]
    if i==1:
        col=(0,255,0) #Yellow Line -> Green
    elif i==2:
        col=(255,0,0) #White Line -> Blue
    else:
        col=(0,0,255) #
    for cnt in contours:
        epsilon = 0.001*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if len(approx>4) and area>1:
            #print(area/perimeter)
            if area/perimeter>0.1:
                pts = np.array(approx, np.int32)
                pts = pts.reshape((-1,1,2))
                image2 = cv2.polylines(image,[pts],True,col)
    cv2_imshow(image2)
    print("Image2 shape: ",image2.shape)

image = cv2.imread(f'{image_folder}/{img_num}.{ext}')

category_names = ["background", "yellow_line","white_line","obstacles","duckiebot","red_line"]
categories = []
for i, name in enumerate(category_names):
    if i!=0:
        categories.append({"supercategory": name, "id":i, "name": name})

print(categories)

import glob
orig_files = glob.glob(f"{image_folder}*.{ext}")
#import random
random.shuffle(orig_files)

files = orig_files
#files = orig_files[0:7000]
#val_files = orig_files[7000:10000]
import os
images = []
basename_to_id = {}
for i, file in enumerate(files):
    basename = os.path.basename(file)
    entry = {"file_name":basename, "height":240, "width":320, "id":i}
    print("entry: ",entry)
    basename_to_id[basename]=entry["id"]
    images.append(entry)
#images
    
ret,thresh = cv2.threshold(image,127,255,0)

print(thresh.shape)

from cv2_plt_imshow import cv2_plt_imshow as cv2_imshow
annotation={}
def get_annotations(image_file_name, mask_file_name, annotation_id):
    global annotation, pts
    image_annotations=[]
    mask_flattened = np.load(mask_file_name)
    if dataset_name == "js":
        mask_flattened = cv2.resize(mask_flattened.reshape(240,320).astype(np.uint8), (320,240), cv2.INTER_NEAREST)

    basename = os.path.basename(image_file_name)
    h, w = 320, 240
    mask_reshaped = mask_flattened.reshape(w,h)
    image = cv2.imread(image_file_name)
    if dataset_name == "js":
        image = cv2.resize(image, (320,240))
    for i in range(1,6):
        thresh = (mask_reshaped==i).astype(np.uint8)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        #cnt = contours[1]
        if real:
            i=real2sim[i]
        if dataset_name=="js":
            i=js2sim[i]
        if i==1:
            col=(0,255,0)
        elif i==2:
            col=(255,0,0)
        else:
            col=(0,0,255)
        for cnt in contours:
            epsilon = 0.002*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if len(approx>4) and area>1:
                #print(area/perimeter)
                if area/perimeter>0.1:
                    pts = np.array(approx, np.int32)
                    pts = pts.reshape((-1,1,2))
                    x, y = pts.reshape(-1,2)[:,0], pts.reshape(-1,2)[:,1]
                    bbox = [float(x.min()), float(y.min()), float(x.max()-x.min()),float(y.max()-y.min())]
                    annotation={"area":area,
                        "iscrowd":0,
                        "image_id": basename_to_id[basename],
                        "segmentation": [pts.flatten().tolist()],
                        "bbox": bbox,
                        "category_id": i,
                        "id":annotation_id
                       }
                    annotation_id+=1
                    image_annotations.append(annotation)
                    #print(annotation)
                    #image2 = cv2.polylines(image,[pts],True,col)
        #cv2_imshow(image2)
        
    #print(annotation_id)
    return image_annotations, annotation_id
#img_num=10205
#img_num="frame25468"
mask_file_name = f'{mask_folder}{img_num}.npy'
image_file_name=  f'{image_folder}{img_num}.{ext}'
annotation_id = 1
image_annotations=get_annotations(image_file_name, mask_file_name, annotation_id)
#image_annotations
import json
def get_annorations_and_write(files, suffix="train"):
    annotations=[]
    from tqdm import tqdm
    annotation_id = 1
    for file in tqdm(files):
        basename = os.path.basename(file)
        mask_file_name = f'{mask_folder}{basename.replace(".jpg",".npy").replace(".png",".npy")}'
        image_file_name=  f'{image_folder}{basename}'
        image_annotations, annotation_id =get_annotations(image_file_name, mask_file_name, annotation_id)
        annotations+=image_annotations
    data={"categories":categories, "images":images,"annotations":annotations}
    with open(f"C:/segmentation_code/Yishu/dataset_js_02/image_2020_12_04_11_25_04/w_bezier/duckie_{dataset_name}_{suffix}.json","w") as f:
        json.dump(data,f)



get_annorations_and_write(files[0:7000],"train")
get_annorations_and_write(files[7000:10000],"val")


