
import tensorflow as tf
import os
import cv2
import numpy as np
import argparse
from model.yolov4 import Yolov4
from model.nms import yolov4_nms
from model.nms import NonMaxSuppression
from utils.preprocess import resize_img
import random
import albumentations as A

tta_transform_1 = A.Compose([
    A.HorizontalFlip(p=1.),
    A.RandomBrightnessContrast(p=0.2),
])
tta_transform_2 = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
])
def parse_args(args):
    parser = argparse.ArgumentParser("test model")
    parser.add_argument('--model-type', default='p5', help="choices=['p5','p6','p7']")
    parser.add_argument('--num-classes', default=12)
    parser.add_argument('--pic-dir',default='images/chess_test_set')
    parser.add_argument('--weight-path', default='checkpoints/scaled_yolov4_best_62_0.982')
    parser.add_argument('--img-size', default=(416, 416))
    parser.add_argument('--TTA', default=True)
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2)
    parser.add_argument('--score-threshold', default=0.5)
    parser.add_argument('--class-names', default='dataset/chess.names')
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    return parser.parse_args(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def detect_batch_img(img,model,args):
    img = img / 255
    pre_nms_decoded_boxes, pre_nms__scores = model(img, training=False)
    pre_nms_decoded_boxes = pre_nms_decoded_boxes.numpy()
    pre_nms__scores = pre_nms__scores.numpy()
    boxes, scores, classes, valid_detections = yolov4_nms(args)(pre_nms_decoded_boxes, pre_nms__scores, args)
    return boxes, scores, classes, valid_detections
    # pre_nms_decoded_boxes,pre_nms__scores = model(img)
    # pre_nms_decoded_boxes = pre_nms_decoded_boxes.numpy()
    # pre_nms__scores = pre_nms__scores.numpy()
    # return pre_nms_decoded_boxes, pre_nms__scores
def tta_nms(boxes,scores,classes,valid_detections,args):
    all_boxes = []
    all_scores = []
    all_classes = []
    batch_index = 0
    valid_boxes = boxes[batch_index][0:valid_detections[batch_index]]
    valid_boxes[:, (0, 2)] = (1.-valid_boxes[:,(2,0)])
    all_boxes.append(valid_boxes)
    all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
    all_classes.append(classes[batch_index][0:valid_detections[batch_index]])
    for batch_index in range(1,boxes.shape[0]):
        all_boxes.append(boxes[batch_index][0:valid_detections[batch_index]])
        all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
        all_classes.append(classes[batch_index][0:valid_detections[batch_index]])
    all_boxes = np.concatenate(all_boxes,axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)
    all_boxes,all_scores,all_classes = np.array(all_boxes), np.array(all_scores), np.array(all_classes)
    boxes, scores, classes, valid_detections = NonMaxSuppression.diou_nms_np_tta(np.expand_dims(all_boxes,0),np.expand_dims(all_scores,0),np.expand_dims(all_classes,0),args)
    boxes, scores, classes, valid_detections = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), np.squeeze(valid_detections)
    return boxes[:valid_detections], scores[:valid_detections], classes[:valid_detections]


def plot_one_box(img, box, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 7, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 7, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_boxes(img,boxes,scores,classes,class_names,args):
    for i in range(len(boxes)):
            if scores[i] < args.score_threshold:
                continue
            x1y1 = (boxes[i][0:2] * img.shape[0:2][::-1]).astype(np.int)
            x2y2 = (boxes[i][2:4] * img.shape[0:2][::-1]).astype(np.int)
            plot_one_box(img,[x1y1[0],x1y1[1],x2y2[0],x2y2[1]],(255,0,255),label=str(class_names[classes[i]]) + "," + str("%0.2f" % scores[i]))
            # cv2.rectangle(img, tuple(x1y1), tuple(x2y2), (0, 255, 0), 2)
            # cv2.putText(img, str(class_names[classes[i]]) + " " + str("%0.2f" % scores[i]),
            #             (x1y1[0], max(x1y1[1] - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
def main(args):
    model = Yolov4(args, training=False)
    model.load_weights(args.weight_path)
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    img_list = os.listdir(args.pic_dir)

    for img_name in img_list:
        img = cv2.imread(os.path.join(args.pic_dir, img_name))
        # img_ori = cv2.resize(img, args.img_size, interpolation=cv2.INTER_CUBIC)
        img_ori,_,_ = resize_img(img, args.img_size)
        img_copy = img_ori.copy()
        aug_imgs = []
        if args.TTA:
            aug_imgs.append(tta_transform_1(image=img_copy)['image'])
            aug_imgs.append(tta_transform_2(image=img_copy)['image'])
        aug_imgs.append(img_copy)
        batch_img = np.array(aug_imgs)
        boxes,scores,classes,valid_detections = detect_batch_img(batch_img, model,args)
        boxes, scores, classes = tta_nms(boxes, scores, classes,valid_detections,args)
        plot_boxes(img_copy,boxes,scores,classes,class_names,args)
        cv2.imshow("d", img_copy/255)
        cv2.waitKey(0)
import sys
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
