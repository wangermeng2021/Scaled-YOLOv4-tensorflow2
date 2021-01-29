

import tensorflow as tf
from model.box_coder import box_decode
from model.nms import NonMaxSuppression
from model.nms import yolov4_nms
import sys
def postprocess(outputs, args):

    num_classes = args.num_classes
    if num_classes == 1:
        num_classes = 0

    boxes_list = []
    scores_list = []
    for index, output in enumerate(outputs):
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], -1,
                                     5 + num_classes])
        output = tf.sigmoid(output)
        # print("oooooooooo:", tf.reduce_max(output[..., 4]))
        decoded_boxes = box_decode(output[..., 0:4], args, index)
        if num_classes == 0:
            scores = output[..., 4:5]
        else:
            scores = output[..., 4:5] * output[..., 5:]

        # tf.print("tensors_1:", output[...,4][output[...,4] >0.5], output_stream=sys.stdout)
        # print(output[...,4][output[...,4] >0.5])
        # print(tf.reduce_max(scores,axis=-1)[tf.reduce_max(scores,axis=-1)>0.8])

        # if tf.shape(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])[0]>0:
        #     print("jjjjjjjjjjjjjjjjjjjjjj")
        #     print(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])
        #     print(decoded_boxes[])
        scores = tf.reshape(scores, [tf.shape(scores)[0], -1, tf.shape(scores)[-1]])

        # print(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.5])
        # print(tf.argmax(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.5])
        # print(decoded_boxes[tf.reduce_max(scores, axis=-1) > 0.5])
        # tf.print("tensors_2:", tf.argmax(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.5], output_stream=sys.stdout)

        # print(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.5])
        # print(decoded_boxes[tf.reduce_max(scores, axis=-1) > 0.5])
        # print("jjjjjjjjjjjjjjjjjjjjjj")
        # print(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])
        # print(tf.shape(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])[0])
        # if tf.shape(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])[0]>0:
        #     print("jjjjjjjjjjjjjjjjjjjjjj")
        #     print(tf.reduce_max(scores, axis=-1)[tf.reduce_max(scores, axis=-1) > 0.8])
        #     print(decoded_boxes[tf.reduce_max(scores, axis=-1) > 0.8])


        boxes_list.append(decoded_boxes)
        scores_list.append(scores)
    decoded_boxes = tf.concat(boxes_list, axis=-2)
    scores = tf.concat(scores_list, axis=-2)
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






#
# #test
# import numpy as np
# a1 = np.array([[1,2,3],[5,61,7]])
# a21,a22 = tf.math.top_k(a1,2)
#
# b1 = np.array([[[1,2],[3,4],[5,6]],
#                [[11,22],[33,44],[55,66]]])
#
#
# print(b1)
# print(a22)
# print(tf.gather(b1,a22, batch_dims=1))
# exit()
# indices = [[1], [0]]
# params = [[[1, 2],[11, 22]], [[3, 4],[5, 6]]]
#
# s1 = tf.gather_nd(params,indices)
# print(s1)
#
#
# exit()
#
# b1 = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],
#                [[8,8,8,8],[7,7,7,7],[6,6,6,6]]])
#
# a22 = a22.numpy()
# print(a22)
# # a22 =np.expand_dims(a22,axis=-1)
# # print(a22)
# # print(b1)
# # print(b1[a22])
# # print(b1.shape)
# # print(a22.shape)
# print(b1.shape)
# print(a22.shape)
# # print(a22)