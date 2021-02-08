
import os
import logging
import time
import warnings
import numpy as np
import cv2
from PIL import Image


def preprocess(img, boxes, style=0):
    
    if style == 0:
        img = img/255.
    elif style == 1:#tensorflow
        img = img / 127.5 - 1.0
    else:#caffe,bgr
        img -= [123.68,116.779,103.939]
        
    boxes[..., 0:4] /= np.tile(img[0].shape[0:2][::-1], [2])
    return img.copy(), boxes.copy()

def resize_img_aug(img,dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size/img_wh
    min_scale = np.min(scale)
    random_resize_style = np.random.randint(0, 5)
    resize_list = [cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_LANCZOS4]
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale, interpolation=resize_list[random_resize_style])
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size//2
    img = np.pad(img,[(half_pad_size[1],pad_size[1]-half_pad_size[1]),(half_pad_size[0],pad_size[0]-half_pad_size[0]),(0,0)], constant_values=np.random.randint(0, 255))
    return img, min_scale, pad_size
def resize_img(img,dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size/img_wh
    min_scale = np.min(scale)
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale)
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size//2
    img = np.pad(img,[(half_pad_size[1],pad_size[1]-half_pad_size[1]),(half_pad_size[0],pad_size[0]-half_pad_size[0]),(0,0)])
    return img, min_scale, pad_size
