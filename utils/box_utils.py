
"""

"""
import numpy as np
import tensorflow as tf


def box_iou(box1, box2):

    inter_w = tf.maximum(tf.minimum(box1[..., 2], box2[..., 2]) - tf.maximum(box1[..., 0], box2[..., 0]),0)
    inter_h = tf.maximum(tf.minimum(box1[..., 3], box2[..., 3]) - tf.maximum(box1[..., 1], box2[..., 1]),0)
    inter_area = inter_w * inter_h
    boxes1_area = (box1[..., 2] - box1[..., 0])*(box1[..., 3] - box1[..., 1])
    boxes2_area = (box2[..., 2] - box2[..., 0])*(box2[..., 3] - box2[..., 1])
    # return tf.where(
    #     tf.equal(inter_area, 0.0),
    #     tf.zeros_like(inter_area), tf.truediv(inter_area, boxes1_area+boxes2_area-inter_area))

    scores = inter_area/(boxes1_area+boxes2_area-inter_area+1e-07)
    # print(scores)
    # scores = tf.where(tf.math.is_inf(scores),tf.zeros_like(scores),scores)
    return scores

# print(box_iou(np.array([[1,2,3,4]]),np.array([[5,6,7,8]])))
#
# exit()

def boxes_iou(boxes1, boxes2):
    boxes2 = np.array(boxes2)
    boxes1 = np.expand_dims(boxes1, -2)
    boxes1_wh = boxes1[..., 2:4] - boxes1[..., 0:2]
    inter_area = np.minimum(boxes1_wh[..., 0], boxes2[..., 0]) * np.minimum(boxes1_wh[..., 1], boxes2[..., 1])
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2[..., 0] * boxes2[..., 1]
    return inter_area/(boxes1_area+boxes2_area-inter_area)
def boxes_iou_all(boxes1, boxes2):
    boxes1 = np.expand_dims(boxes1, -2)
    inter_w = np.maximum(np.minimum(boxes1[..., 2], boxes2[..., 2]) - np.maximum(boxes1[..., 0], boxes2[..., 0]),0)
    inter_h = np.maximum(np.minimum(boxes1[..., 3], boxes2[..., 3]) - np.maximum(boxes1[..., 1], boxes2[..., 1]),0)
    inter_area = inter_w * inter_h
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0])*(boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0])*(boxes2[..., 3] - boxes2[..., 1])
    return inter_area/(boxes1_area+boxes2_area-inter_area)

def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)

    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)