
"""

"""
import numpy as np
import tensorflow as tf

"""
Reads Darknet config and weights and creates Keras model with TF backend.
"""
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import argparse
import numpy as np
import tensorflow as tf
import cv2
from model.yolov4_tiny import Yolov4_tiny
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from model.nms import yolov4_nms
from model.nms import NonMaxSuppression
import random
import albumentations as A
from utils.preprocess import resize_img


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert Darknet weight To tensorflow format.')
    parser.add_argument('--mode', default='test',help="convert weight or test weight")
    parser.add_argument('--model-type', default='tiny')
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
    parser.add_argument('--score-threshold', default=0.5, type=float)
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--darknet-weights-path', help='Path to Darknet weights file.',
                        default='pretrain/yolov4-tiny.weights')
    parser.add_argument('--output-weight-path', help='Path to output tf model file.',
                        default='pretrain/yolov4-tiny')
    parser.add_argument('--test-img-path', help='Path to a image to test a model.',
                        default="images/dog.jpg")

    #test model
    parser.add_argument('--tta', default=True)
    parser.add_argument('--img-size', default=(416, 416))
    parser.add_argument('--class-names', default='dataset/coco.names')
    return parser.parse_args(args)


def convert_weight(args):
    print("model is converting...")
    weights_file = open(args.darknet_weights_path, 'rb')
    major, minor, revision, seen, _ = np.fromfile(weights_file, dtype=np.int32, count=5)
    model = Yolov4_tiny(args, training=False)
    # model.summary()

    backbone_layers_list = []
    max_block_num = 5
    for i in range(1,max_block_num+1):
        j = 1
        while True:
            try:
                layer_name = 'block_{}_{}_batch_normalization'.format(i,j)
                model.get_layer(layer_name)
                backbone_layers_list.append(layer_name)
            except:
                pass
            try:
                layer_name = 'block_{}_{}_conv2d'.format(i,j)
                model.get_layer(layer_name)
                backbone_layers_list.append(layer_name)
            except:
                break
            j+=1

    head_layers_list=[]
    head_num = 2
    for i in range(1,head_num+1):
        j = 1
        while True:
            try:
                layer_name = 'yolov3_head_{}_{}_batch_normalization'.format(i,j)
                model.get_layer(layer_name)
                head_layers_list.append(layer_name)
            except:
                pass
            try:
                layer_name = 'yolov3_head_{}_{}_conv2d'.format(i,j)
                model.get_layer(layer_name)
                head_layers_list.append(layer_name)
            except:
                break
            j+=1

    model_layer_list = backbone_layers_list[0:-2]+head_layers_list[3:]+backbone_layers_list[-2:]+head_layers_list[0:3]
    i=0
    while i<len(model_layer_list):
        layer_name = model_layer_list[i]
        if 'batch_normalization' in layer_name:
            conv_name = layer_name.replace('batch_normalization', 'conv2d')
            filters = model.get_layer(conv_name).filters
            size = model.get_layer(conv_name).kernel_size[0]
            in_dim = model.get_layer(conv_name).get_input_shape_at(0)[-1]
            # darknet [beta, gamma, mean, variance]
            bn_weights = np.fromfile(
                weights_file, dtype=np.float32, count=4 * filters)
            # tf [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                weights_file, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            model.get_layer(conv_name).set_weights([conv_weights])
            model.get_layer(layer_name).set_weights(bn_weights)
            i+=2

        elif 'conv2d' in layer_name:
            filters = model.get_layer(layer_name).filters
            size = model.get_layer(layer_name).kernel_size[0]
            in_dim = model.get_layer(layer_name).get_input_shape_at(0)[-1]

            conv_bias = np.fromfile(weights_file, dtype=np.float32, count=filters)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                weights_file, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            model.get_layer(layer_name).set_weights([conv_weights, conv_bias])
            i += 1

    assert len(weights_file.read()) == 0, 'failed to read all data'
    weights_file.close()
    model.save(args.output_weight_path)
    # model.save_weights('/home/wangem1/papers_coding/111111/coco_pretrain')
    print("the converting process is finished!")

tta_transform_1 = A.Compose([
    A.HorizontalFlip(p=1.),
    A.RandomBrightnessContrast(p=0.2),
])
tta_transform_2 = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
])
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
def test_model(args):
    model = tf.keras.models.load_model(args.output_weight_path)
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    img = cv2.imread(args.test_img_path)
    img_ori, _, _ = resize_img(img, args.img_size)
    img_copy = img_ori.copy()
    aug_imgs = []
    if args.tta:
        aug_imgs.append(tta_transform_1(image=img_copy)['image'])
        aug_imgs.append(tta_transform_2(image=img_copy)['image'])
    aug_imgs.append(img_copy)
    batch_img = np.array(aug_imgs)
    boxes, scores, classes, valid_detections = detect_batch_img(batch_img, model, args)
    boxes, scores, classes = tta_nms(boxes, scores, classes, valid_detections, args)
    plot_boxes(img_copy, boxes, scores, classes, class_names, args)
    cv2.imshow("d", img_copy / 255)
    cv2.waitKey(0)
import sys
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args.mode=='convert':
        convert_weight(args)
    else:
        test_model(args)
