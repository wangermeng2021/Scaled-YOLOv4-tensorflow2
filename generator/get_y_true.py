"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import cv2
from PIL import Image

from utils import box_utils

from utils.anchors import yolo_anchors

def get_y_true(max_side, batch_boxes, groundtruth_valids, args):


    if args.model_type == 'p5':
        detect_layer_num = 3
    if args.model_type == 'p6':
        detect_layer_num = 4
    if args.model_type == 'p7':
        detect_layer_num = 5
    strides = [8*2**i for i in range(detect_layer_num)]
    offset = 0.5

    class_num = args.num_classes
    batch_size = batch_boxes.shape[0]
    grid_size = max_side//np.array(strides).astype(np.int32)

    if args.anchor_match_type == 'iou':
        iou_scores = box_utils.boxes_iou(batch_boxes, np.reshape(np.array(yolo_anchors[args.model_type]), [-1,2])/(max_side, max_side))
        matched_anchor_index = np.argsort(-iou_scores,axis=-1)
        matched_anchor_num = np.sum(iou_scores>args.anchor_match_iou_thr, axis=-1)
        if args.anchor_match_iou_thr == -1:
            matched_anchor_num = np.ones_like(matched_anchor_num)
    else:
        anchors_wh = np.reshape(np.array(yolo_anchors[args.model_type]), [-1, 2]) / (max_side, max_side)
        batch_boxes_wh = batch_boxes[..., 2:4] - batch_boxes[..., 0:2]
        batch_boxes_wh = batch_boxes_wh[:,:, None]
        wh_ratio = anchors_wh/(batch_boxes_wh+1e-7)
        wh_ratio = np.max(np.maximum(wh_ratio, 1./wh_ratio),axis=-1)
        matched_anchor_index = np.argsort(wh_ratio, axis=-1)
        matched_anchor_num = np.sum(wh_ratio < args.anchor_match_wh_ratio_thr, axis=-1)

    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    batch_boxes = np.concatenate([batch_boxes, matched_anchor_index,matched_anchor_num], axis=-1)


    grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
    grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
    grids_2 = np.zeros([batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
    grids = [grids_0, grids_1, grids_2]

    for batch_index in range(batch_size):
        for box_index in range(groundtruth_valids[batch_index]):
            for anchor_index in batch_boxes[batch_index][box_index][5:5 + int(batch_boxes[batch_index][box_index][-1])]:
                grid_index = (anchor_index // len(yolo_anchors[args.model_type][0])).astype(np.int32)
                grid_anchor_index = (anchor_index % len(yolo_anchors[args.model_type][0])).astype(np.int32)
                cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2
                cxcy *= grid_size[grid_index]
                grid_xy = np.floor(cxcy).astype(np.int32)

                dxdy = cxcy - grid_xy
                # dwdh = batch_boxes[batch_index][box_index][2:4]/np.array(strides[grid_index])
                dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                dwdh = dwdh*np.array(grid_size[grid_index])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][5+batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                # continue
                # print(grid_xy[1], grid_xy[0],anchor_index)
                grid_xy_fract = cxcy%1.
                if (grid_xy > 0).all():
                    if grid_xy_fract[0] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                    if grid_xy_fract[1] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                if (grid_xy<grid_size[grid_index]-1).all():
                    if grid_xy_fract[0] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                    if grid_xy_fract[1] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][
                            5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

    return tuple(grids)


def get_y_true_with_one_class(max_side, batch_boxes, groundtruth_valids, args):
    if args.model_type == 'p5':
        detect_layer_num = 3
    if args.model_type == 'p6':
        detect_layer_num = 4
    if args.model_type == 'p7':
        detect_layer_num = 5
    strides = [8*2**i for i in range(detect_layer_num)]
    offset = 0.5
    # class_num = args.num_classes
    class_num = 0
    batch_size = batch_boxes.shape[0]
    grid_size = max_side//np.array(strides).astype(np.int32)

    if args.anchor_match_type == 'iou':
        iou_scores = box_utils.boxes_iou(batch_boxes, np.reshape(np.array(yolo_anchors[args.model_type]), [-1,2])/(max_side, max_side))
        matched_anchor_index = np.argsort(-iou_scores,axis=-1)
        matched_anchor_num = np.sum(iou_scores>args.anchor_match_iou_thr, axis=-1)
        if args.anchor_match_iou_thr == -1:
            matched_anchor_num = np.ones_like(matched_anchor_num)
    else:
        anchors_wh = np.reshape(np.array(yolo_anchors[args.model_type]), [-1, 2]) / (max_side, max_side)
        batch_boxes_wh = batch_boxes[..., 2:4] - batch_boxes[..., 0:2]
        batch_boxes_wh = batch_boxes_wh[:, :,None]
        # print(batch_boxes_wh)
        wh_ratio = anchors_wh/batch_boxes_wh
        wh_ratio = np.max(np.maximum(wh_ratio, 1./wh_ratio),axis=-1)
        matched_anchor_index = np.argsort(wh_ratio, axis=-1)
        matched_anchor_num = np.sum(wh_ratio < args.anchor_match_wh_ratio_thr, axis=-1)

    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    batch_boxes = np.concatenate([batch_boxes, matched_anchor_index,matched_anchor_num], axis=-1)

    if args.model_type=='p5':
        grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros([batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1, grids_2]
    elif args.model_type=='p6':
        grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros([batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_3 = np.zeros([batch_size, grid_size[3], grid_size[3], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1, grids_2, grids_3]
    else :
        grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros([batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_3 = np.zeros([batch_size, grid_size[3], grid_size[3], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_4 = np.zeros([batch_size, grid_size[4], grid_size[4], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1, grids_2, grids_3, grids_4]

    for batch_index in range(batch_size):
        for box_index in range(groundtruth_valids[batch_index]):
            for anchor_index in batch_boxes[batch_index][box_index][5:5 + int(batch_boxes[batch_index][box_index][-1])]:
                grid_index = (anchor_index // len(yolo_anchors[args.model_type][0])).astype(np.int32)
                grid_anchor_index = (anchor_index % len(yolo_anchors[args.model_type][0])).astype(np.int32)
                cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2
                cxcy *= grid_size[grid_index]
                grid_xy = np.floor(cxcy).astype(np.int32)

                dxdy = cxcy - grid_xy
                # dwdh = batch_boxes[batch_index][box_index][2:4]/np.array(strides[grid_index])
                dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                dwdh = dwdh*np.array(grid_size[grid_index])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                # grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][5+batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                # continue
                # print(grid_xy[1], grid_xy[0],anchor_index)
                grid_xy_fract = cxcy%1.
                if (grid_xy > 0).all():
                    if grid_xy_fract[0] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1
                        # grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][
                        #     5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                    if grid_xy_fract[1] < offset:
                        dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1
                        # grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][
                        #     5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

                if (grid_xy<grid_size[grid_index]-1).all():
                    if grid_xy_fract[0] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1
                        # grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][
                        #     5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1
                    if grid_xy_fract[1] > offset:
                        dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                        # dwdh = batch_boxes[batch_index][box_index][2:4] / np.array(strides[grid_index])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1
                        # grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][
                        #     5 + batch_boxes[batch_index][box_index][4].astype(np.int32)] = 1

    return tuple(grids)


#
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# def parse_args(args):
#     parser = argparse.ArgumentParser(description='Simple training script for using snapmix .')
#     parser.add_argument('--model-type', default='p5', help="choices=['p5','p6','p7']")
#     parser.add_argument('--reg-losss-weight', default=1.0)
#     parser.add_argument('--obj-losss-weight', default=1.0)
#     parser.add_argument('--cls-losss-weight', default=1.0)
#
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--batch-size', default=16, type=int)
#
#     parser.add_argument('--dataset-root-dir', default='/home/wangem1/dataset/VOC2007&2012', type=str, help="voc,coco")
#     parser.add_argument('--dataset_type', default='voc', type=str, help="voc,coco")
#     parser.add_argument('--voc_train_set', default=[(2007, 'trainval'), (2012, 'trainval')])
#     parser.add_argument('--voc_valid_set', default=[(2007, 'test')])
#     parser.add_argument('--voc_skip_difficult', default=True)
#     parser.add_argument('--coco_train_set', default='train')
#     parser.add_argument('--coco_valid_set', default='valid')
#     parser.add_argument('--num-classes', default=80, help="choices=['p5','p6','p7']")
#     parser.add_argument('--class_names', default='voc.names', help="choices=['p5','p6','p7']")
#
#     parser.add_argument('--augment', default='rand_augment', type=str, help="choices=[random_crop,'mosaic','only_flip_left_right',None]")
#
#     parser.add_argument('--multi-scale', default=[416], help="choices=['p5','p6','p7']")
#
#     parser.add_argument('--start-val-epoch', default=50, type=int)
#
#     parser.add_argument('--optimizer', default='adam', help="choices=[adam,sgd]")
#     parser.add_argument('--momentum', default=0.9, help="choices=[sgd,'p6','p7']")
#     parser.add_argument('--nesterov', default=True, help="choices=[sgd,'p6','p7']")
#     parser.add_argument('--weight_decay', default=True, help="")
#
#     parser.add_argument('--lr-scheduler', default='step', type=str, help="choices=['step','warmup_cosinedecay']")
#     parser.add_argument('--init-lr', default=1e-3, type=float)
#     parser.add_argument('--lr-decay', default=0.1, type=float)
#     parser.add_argument('--lr-decay-epoch', default=[80, 150, 180], type=int)
#     parser.add_argument('--warmup-lr', default=1e-4, type=float)
#     parser.add_argument('--warmup-epochs', default=0, type=int)
#     parser.add_argument('--weight-decay', default=1e-4, type=float)
#
#     parser.add_argument('--transfer-type', default='csp_darknet53_and_pan', help="choices=['p5','p6','p7']")#choices=['csp_darknet53','csp_darknet53_and_pan',None]
#     parser.add_argument('--pretrained-weights', default='p5', help="choices=['p5','p6','p7']")
#     parser.add_argument('--output-checkpoints-dir', default='p5', help="choices=['p5','p6','p7']")
#
#     #los
#     parser.add_argument('--box-regression-loss', default='diou')#{'giou','diou','ciou'}
#     parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")#,#
#     parser.add_argument('--object-loss', default='bce', help="choices=['p5','p6','p7']")
#     parser.add_argument('--focal-alpha', default= 0.25, help="choices=['p5','p6','p7']")
#     parser.add_argument('--focal-gamma', default=2.0, help="choices=['p5','p6','p7']")
#     parser.add_argument('--ignore-thr', default=0.7, help="choices=['p5','p6','p7']")
#
#     #postprocess
#     parser.add_argument('--nms', default=[416], help="choices=['p5','p6','p7']")
#     parser.add_argument('--max-boxes-num', default=1000, help="choices=['p5','p6','p7']")
#     parser.add_argument('--nms-iou-threshold', default=1000, help="choices=['p5','p6','p7']")
#     parser.add_argument('--score-threshold', default=[416], help="choices=['p5','p6','p7']")
#     parser.add_argument('--pre-nms-num-boxes', default=1000, help="choices=['p5','p6','p7']")
#
#     parser.add_argument('--label-smooth', default=0.0)
#     parser.add_argument('--scales-x-y', default=[2., 2., 2.], help="choices=['p5','p6','p7']")
#
#     parser.add_argument('--anchor-match-type', default='iou',help="choices=['iou','wh_ratio']")
#     parser.add_argument('--anchor-match-iou_thr', default=0.5, help="choices=['p5','p6','p7']")
#     parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0)
#
#     return parser.parse_args(args)

# import argparse
# import sys
# if __name__ == "__main__":
#     args = parse_args(sys.argv[1:])
# # #test#
# # from config.param_config import CFG
# batch_boxes = np.array([[
#     # [0.1,0.1,0.2,0.2,1],
#     # [0.1,0.4,0.2,0.5,2],
#     [0.13,0.13,0.96,0.9,3]]])
# groundtruth_valids = np.array([1])
# #get_y_true_with_one_class
# o1 = get_y_true(618, batch_boxes, groundtruth_valids,args)
# s1 = o1[0][...,4]==1
# print(len(o1[0][s1]))
# s1 = o1[1][...,4]==1
# print(len(o1[1][s1]))
# s1 = o1[2][...,4]==1
# print(len(o1[2][s1]))
# # # print(o1)