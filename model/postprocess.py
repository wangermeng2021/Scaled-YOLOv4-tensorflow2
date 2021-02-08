

import tensorflow as tf
from model.box_coder import box_decode
from model.nms import NonMaxSuppression
from model.nms import yolov4_nms
import sys


def postprocess(outputs, args):

    num_classes = int(args.num_classes)
    if num_classes == 1:
        num_classes = 0

    boxes_list = []
    scores_list = []
    for index, output in enumerate(outputs):
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], -1,
                                     5 + num_classes])
        output = tf.sigmoid(output)
        decoded_boxes = box_decode(output[..., 0:4], args, index)
        if num_classes == 0:
            scores = output[..., 4:5]
        else:
            scores = output[..., 4:5] * output[..., 5:]


        scores = tf.reshape(scores, [tf.shape(scores)[0], -1, tf.shape(scores)[-1]])

        boxes_list.append(decoded_boxes)
        scores_list.append(scores)
    decoded_boxes = tf.concat(boxes_list, axis=-2,name='output_boxes')
    scores = tf.concat(scores_list, axis=-2,name='output_scores')
    return decoded_boxes, scores
    max_scores = tf.math.reduce_max(scores, axis=-1)
    pre_nms_values, pre_nms__indices = tf.math.top_k(max_scores, args.pre_nms_num_boxes)
    pre_nms_decoded_boxes = tf.gather(decoded_boxes, pre_nms__indices, batch_dims=1)
    pre_nms__scores = tf.gather(scores, pre_nms__indices, batch_dims=1)

    # boxes, scores, classes, valid_detections = NonMaxSuppression.hard_nms_tf(pre_nms_decoded_boxes, pre_nms__scores, params)
    # return boxes, scores, classes, valid_detections
    #
    return pre_nms_decoded_boxes,pre_nms__scores
    # boxes, scores, classes, valid_detections = yolov4_nms(params)(decoded_boxes,scores,params)
    # # boxes, scores, classes, valid_detections = NonMaxSuppression.hard_nms(decoded_boxes, scores, params)
    # return boxes, scores, classes, valid_detections


