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

    if args.model_type == 'tiny':
        detect_layer_num = 2
        strides = [16 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p5':
        detect_layer_num = 3
        strides = [8 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p6':
        detect_layer_num = 4
        strides = [8 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p7':
        detect_layer_num = 5
        strides = [8 * 2 ** i for i in range(detect_layer_num)]

    offset = 0.5

    class_num = int(args.num_classes)
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

    if args.model_type=='tiny':
        grids_0 = np.zeros([batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros([batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1]
    elif args.model_type=='p5':
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
    if args.model_type == 'tiny':
        detect_layer_num = 2
        strides = [16 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p5':
        detect_layer_num = 3
        strides = [8 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p6':
        detect_layer_num = 4
        strides = [8 * 2 ** i for i in range(detect_layer_num)]
    if args.model_type == 'p7':
        detect_layer_num = 5
        strides = [8 * 2 ** i for i in range(detect_layer_num)]

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
        wh_ratio = anchors_wh/(batch_boxes_wh+1e-7)
        wh_ratio = np.max(np.maximum(wh_ratio, 1./wh_ratio),axis=-1)
        matched_anchor_index = np.argsort(wh_ratio, axis=-1)
        matched_anchor_num = np.sum(wh_ratio < args.anchor_match_wh_ratio_thr, axis=-1)

    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    batch_boxes = np.concatenate([batch_boxes, matched_anchor_index,matched_anchor_num], axis=-1)

    if args.model_type == 'tiny':
        grids_0 = np.zeros(
            [batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros(
            [batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1]
    elif args.model_type == 'p5':
        grids_0 = np.zeros(
            [batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros(
            [batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros(
            [batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1, grids_2]
    elif args.model_type == 'p6':
        grids_0 = np.zeros(
            [batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros(
            [batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros(
            [batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_3 = np.zeros(
            [batch_size, grid_size[3], grid_size[3], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids = [grids_0, grids_1, grids_2, grids_3]
    else:
        grids_0 = np.zeros(
            [batch_size, grid_size[0], grid_size[0], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_1 = np.zeros(
            [batch_size, grid_size[1], grid_size[1], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_2 = np.zeros(
            [batch_size, grid_size[2], grid_size[2], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_3 = np.zeros(
            [batch_size, grid_size[3], grid_size[3], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
        grids_4 = np.zeros(
            [batch_size, grid_size[4], grid_size[4], len(yolo_anchors[args.model_type][0]), 5 + class_num], np.float32)
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

