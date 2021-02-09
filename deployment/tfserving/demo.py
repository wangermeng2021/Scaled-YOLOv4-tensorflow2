
import os
import argparse
import sys
import cv2
import numpy as np
import time
from tfservingclient.client import Client
def parse_args(args):
    parser = argparse.ArgumentParser("test model")
    parser.add_argument('--pic-dir',default="../../images/pothole_pictures")
    parser.add_argument('--class-names',default="../../dataset/pothole.names")
    return parser.parse_args(args)
def main(args):
    if not os.path.exists(args.pic_dir):
        raise ValueError("{} don't exist!".format(args.pic_dir))
    if not os.path.exists(args.class_names):
        raise ValueError("{} don't exist!".format(args.class_names))
    with open(args.class_names) as f1:
        class_names = f1.read().splitlines()

    client = Client()
    client.init(host='127.0.0.1',port=8500)
    while True:
        for img_name in os.listdir(args.pic_dir):
            img = cv2.imread(os.path.join(args.pic_dir,img_name))
            img = np.expand_dims(img, axis=0)
            img = client.preprocess(img,(416,416))
            boxes, scores, classes, valid_detections = client.predict(img,score_thr=0.1)
            for index, num_det in enumerate(valid_detections):
                show_img = client.draw_result(img[index], boxes[index][0:num_det], scores[index][0:num_det],
                                              classes[index][0:num_det],class_names)
                cv2.imshow('dd', show_img)
                cv2.waitKey(0)
if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)
