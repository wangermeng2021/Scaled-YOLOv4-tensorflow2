
import tensorflow as tf
from model.common import conv2d_bn_mish

from utils.box_utils import broadcast_iou
from utils.box_utils import box_iou
from utils.anchors import yolo_anchors
import sys
import math
class Yolov3BoxRegressionLoss():

    @staticmethod
    def ciou(args, grid_index, type):
        def loss(y_true, y_pred):

            obj_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(obj_mask,axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)

            grid_height = tf.shape(y_true)[1]
            grid_width = tf.shape(y_true)[2]
            grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
            grid_xy = tf.reshape(grid_xy, [tf.shape(grid_xy)[0], tf.shape(grid_xy)[1], 1, tf.shape(grid_xy)[2]])
            grid_xy = tf.cast(grid_xy, tf.dtypes.float32)

            y_true_cxcy = grid_xy+y_true[...,0:2]
            y_true_x1y1x2y2 = tf.concat([y_true_cxcy - y_true[..., 2:4] / 2,y_true_cxcy + y_true[..., 2:4] / 2],axis=-1)

            scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
            scaled_pred_xy = tf.sigmoid(y_pred[..., 0:2]) * scales_x_y - 0.5 * (scales_x_y - 1)
            # pred_xy = (grid_xy + scaled_pred_xy)/(grid_width, grid_height)
            pred_xy = (grid_xy + scaled_pred_xy)

            # normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
            #     8 * (2 ** grid_index) *tf.shape(y_true)[1:3][::-1], tf.dtypes.float32)
            # pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors
            normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
                8 * (2 ** grid_index), tf.dtypes.float32)
            pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors

            # return tf.reduce_sum(tf.abs(scaled_pred_xy - y_true[..., 0:2]) + tf.abs(pred_wh - y_true[..., 2:4]),[1,2,3,4])/pos_num,1

            pred_x1y1x2y2 = tf.concat([pred_xy-pred_wh/2, pred_xy+pred_wh/2], axis=-1)
            # # pred_x1y1x2y2 = tf.clip_by_value(pred_x1y1x2y2,0,1)
            # print("ssssssssssssssssssss:")
            # print(pred_x1y1x2y2[y_true[..., 4]>0])
            # # print(y_pred[..., 5:][y_true[..., 4] > 0])
            # print(tf.sigmoid(y_pred[..., 4][y_true[..., 4] > 0]))
            # print("pred_scale",((tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2)[y_true[..., 4] > 0] )
            # print("pred_wh",pred_wh[y_true[..., 4] > 0],normalized_anchors)

            # print("sig_max:",
            #       tf.reduce_max(tf.sigmoid(y_pred[..., 4])))
            # print("sig_num:",tf.reduce_sum(tf.cast(tf.sigmoid(y_pred[..., 4])>0.001,tf.dtypes.float32)))
            # # print(y_pred[..., 4:5][y_true[..., 4] > 0])
            # print(tf.sigmoid(y_pred[..., 4][y_true[..., 4] > 0]))
            # print(tf.reduce_sum(tf.cast(tf.sigmoid(y_pred[..., 4])>0.5,tf.dtypes.float32)))
            # # return tf.reduce_sum(
            # #     obj_mask * tf.reduce_sum(tf.square(y_true_x1y1x2y2 - pred_x1y1x2y2), axis=-1),
            # #     axis=[1, 2, 3])/pos_num, 1


            #iou
            inter_w = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 2], y_true_x1y1x2y2[..., 2]) - tf.maximum(pred_x1y1x2y2[..., 0], y_true_x1y1x2y2[..., 0]), 0)
            inter_h = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 3], y_true_x1y1x2y2[..., 3]) - tf.maximum(pred_x1y1x2y2[..., 1], y_true_x1y1x2y2[..., 1]), 0)
            inter_area = inter_w * inter_h
            pred_wh = pred_x1y1x2y2[..., 2:4] - pred_x1y1x2y2[..., 0:2]
            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            boxes1_area = pred_wh[...,0] * pred_wh[...,1]
            boxes2_area = y_true_wh[..., 0] * y_true_wh[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area + 1e-07

            iou_scores = inter_area / union_area

            box_center_dist = (pred_x1y1x2y2[..., 0:2] + pred_x1y1x2y2[..., 2:4])/2-(y_true_x1y1x2y2[..., 0:2] + y_true_x1y1x2y2[..., 2:4]) / 2
            box_center_dist = tf.reduce_sum(tf.math.square(box_center_dist), axis=-1)
            bounding_rect_wh = tf.maximum(pred_x1y1x2y2[..., 2:4], y_true_x1y1x2y2[..., 2:4])-tf.minimum(pred_x1y1x2y2[..., 0:2], y_true_x1y1x2y2[..., 0:2])
            diagonal_line_length = tf.reduce_sum(tf.math.square(bounding_rect_wh),axis=-1)
            diou_loss = iou_scores - box_center_dist/(diagonal_line_length+1e-07)

            # return tf.reduce_sum(obj_mask * (1 - iou_scores), [1, 2, 3]) / pos_num,iou_scores

            # type = 'giou'
            if type == 'giou':
                bounding_rect_area = bounding_rect_wh[..., 0] * bounding_rect_wh[...,1] + 1e-07
                giou_loss = iou_scores - (bounding_rect_area-union_area)/bounding_rect_area
                return tf.reduce_sum(obj_mask*(1-giou_loss),[1,2,3])/pos_num, giou_loss
            if type == 'diou':
                return tf.reduce_sum(obj_mask*(1-diou_loss),[1,2,3])/pos_num, diou_loss

            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            v = (4./math.pi**2)*tf.square(tf.math.atan2(pred_wh[..., 1], pred_wh[...,0])-tf.math.atan2(y_true_wh[..., 1], y_true_wh[...,0]))

            alpha = v/(1.-iou_scores+v+1e-07)

            # alpha = v * tf.stop_gradient(pred_wh[..., 0] * pred_wh[..., 0], pred_wh[..., 1] * pred_wh[..., 1])
            alpha = tf.stop_gradient(alpha)
            ciou_loss = diou_loss - alpha*v
            return tf.reduce_sum(obj_mask*(1 - ciou_loss), [1,2,3])/pos_num,ciou_loss
        return loss

class Yolov3ClassificationLoss():


    @staticmethod
    def bce(args, grid_index):
        def loss(y_true, y_pred):
            object_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(object_mask, axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)

            # y_true = y_true[..., 5:]
            # y_pred = y_pred[..., 5:]

            # classification_loss = object_mask*tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=args.label_smooth,reduction='none')(y_true[..., 5:], y_pred[..., 5:])
            classification_loss = object_mask * tf.keras.losses.binary_crossentropy(y_true[..., 5:],y_pred[..., 5:],from_logits=True,label_smoothing=args.label_smooth)

            return tf.reduce_sum(classification_loss, axis=[1,2,3])/pos_num
        return loss

    @staticmethod
    def focal(args, grid_index):
        def loss(y_true, y_pred):
            object_mask = y_true[..., 4]
            y_true = y_true[..., 5:]
            y_pred = y_pred[..., 5:]
            y_pred = tf.math.sigmoid(y_pred)
            alpha_weight = tf.where(y_true != 0, args.focal_alpha, 1.-args.focal_alpha)
            focal_weight = tf.where(y_true != 0, 1. - y_pred, y_pred)
            focal_weight = alpha_weight * focal_weight ** args.focal_gamma
            bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            focal_loss = focal_weight*bce_loss
            focal_loss = tf.reduce_sum(focal_loss, axis=[-1])
            focal_loss = tf.reduce_sum(object_mask*focal_loss,axis=[1, 2, 3])
            normalizer = tf.maximum(tf.reduce_sum(object_mask, axis=[1, 2, 3]), 1.0)
            return focal_loss/normalizer
        return loss
import numpy as np
class Yolov3ObjectLoss():
    @staticmethod
    def bce(args, grid_index):
        def loss(y_true, y_pred, iou_score):

            object_loss_mask = y_true[..., 4]
            # object_loss_ori = tf.keras.losses.binary_crossentropy(y_true[..., 4:5], y_pred[..., 4:5], from_logits=True)
            iou_score = tf.maximum(iou_score, 0)
            # print("nnnnnnnnnnnnnnnnnn:")
            # print(iou_score[iou_score>0.5])
            # print(len(iou_score[iou_score>0.5]))
            iou_score = tf.stop_gradient(iou_score)
            object_loss_ori = tf.keras.losses.binary_crossentropy(tf.expand_dims(iou_score,axis=-1),
                                                                  y_pred[..., 4:5], from_logits=True)
            return tf.reduce_mean(object_loss_ori, axis=[1, 2, 3])

            # grid_height = tf.shape(y_true)[1]
            # grid_width = tf.shape(y_true)[2]
            # grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
            # grid_xy = tf.reshape(grid_xy, [tf.shape(grid_xy)[0], tf.shape(grid_xy)[1], 1, tf.shape(grid_xy)[2]])
            # grid_xy = tf.cast(grid_xy, tf.dtypes.float32)
            # y_true_cxcy = grid_xy+y_true[...,0:2]
            # y_true_x1y1x2y2 = tf.concat([y_true_cxcy - y_true[..., 2:4] / 2,y_true_cxcy + y_true[..., 2:4] / 2],axis=-1)
            # scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
            # scaled_pred_xy = tf.sigmoid(y_pred[..., 0:2]) * scales_x_y - 0.5 * (scales_x_y - 1)
            # pred_xy = (grid_xy + scaled_pred_xy)
            # normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
            #     8 * (2 ** grid_index), tf.dtypes.float32)
            # pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors
            # pred_x1y1x2y2 = tf.concat([pred_xy-pred_wh/2, pred_xy+pred_wh/2], axis=-1)
            #
            # iou_scores = broadcast_iou(pred_x1y1x2y2, tf.boolean_mask(y_true_x1y1x2y2, object_loss_mask))
            # iou_max_value = tf.reduce_max(iou_scores, axis=-1)
            # object_loss_ignore_mask = tf.cast(iou_max_value < args.ignore_thr, tf.dtypes.float32)
            # object_loss = object_loss_mask*object_loss_ori+(1-object_loss_mask)*object_loss_ignore_mask*object_loss_ori
            # num_valid = tf.reduce_sum(object_loss_mask,axis=[1,2,3])+tf.reduce_sum((1-object_loss_mask)*object_loss_ignore_mask,axis=[1,2,3])
            # num_valid = tf.maximum(num_valid,1)
            # return tf.reduce_sum(object_loss, axis=[1,2,3])/num_valid
        return loss

def yolov3_loss(args, grid_index):


    if args.box_regression_loss == 'diou':
        box_regression_loss = Yolov3BoxRegressionLoss.ciou(args, grid_index, 'diou')
    elif args.box_regression_loss == 'ciou':
        box_regression_loss = Yolov3BoxRegressionLoss.ciou(args, grid_index, 'ciou')
    elif args.box_regression_loss == 'giou':
        box_regression_loss = Yolov3BoxRegressionLoss.ciou(args, grid_index, 'giou')

    if args.classification_loss == 'bce':
        classification_loss = Yolov3ClassificationLoss.bce(args, grid_index)
    elif args.classification_loss == 'focal':
        classification_loss = Yolov3ClassificationLoss.focal(args, grid_index)


    object_loss = Yolov3ObjectLoss.bce(args, grid_index)

    def loss(y_true, y_pred):

        model_obj_loss_layers_weights={"p5":[4.0, 1.0, 0.4],
                       "p6": [4.0, 1.0, 0.4, 0.1],
                       "p7": [4.0, 1.0, 0.5, 0.4, 0.1]}
        obj_loss_layers_weights = model_obj_loss_layers_weights[args.model_type][grid_index]

        detect_layer_num = len(model_obj_loss_layers_weights[args.model_type])
        model_loss_scale = 3 / detect_layer_num

        box_reg_losss_weight = args.reg_losss_weight * model_loss_scale
        obj_losss_weight = args.obj_losss_weight * model_loss_scale * (1.4 if detect_layer_num >= 4 else 1.)
        cls_losss_weight = args.cls_losss_weight * model_loss_scale

        box_reg_loss, iou_score = box_regression_loss(y_true, y_pred)
        obj_loss = object_loss(y_true, y_pred, iou_score)*obj_loss_layers_weights
        cls_loss = classification_loss(y_true, y_pred)

        #
        # return box_reg_loss+obj_loss
        if args.num_classes == 1:
            # print("\n")
            # print("loss1:", box_reg_loss)
            # print("loss2:", obj_loss)
            # print("loss3:", obj_loss_layers_weights)
            return box_reg_losss_weight*box_reg_loss + obj_losss_weight*obj_loss
        return  box_reg_losss_weight*box_reg_loss + obj_losss_weight*obj_loss + cls_losss_weight*cls_loss

    return loss

#
# import os
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
#     parser.add_argument('--box-regression-loss', default='giou')#{'giou','diou','ciou'}
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
#
# import argparse
# import sys
# if __name__ == "__main__":
#     args = parse_args(sys.argv[1:])
#
#     while True:
#         a1 = np.random.uniform(0, 0.4, size=[1, 52, 52, 4, 6]).astype(np.float32)
#         a2 = np.random.uniform(0, 0.4, size=[1, 52, 52, 4, 6]).astype(np.float32)
#         loss1 = yolov3_loss(args,0)(a1,a2)
#         loss2 = yolov3_loss(args, 1)(a1, a2)
#         loss3 = yolov3_loss(args, 2)(a1, a2)
#         print(tf.reduce_sum(loss1))
#         print(tf.reduce_sum(loss2))
#         print(tf.reduce_sum(loss3))
#
#
#

# from config.param_config import CFG
# import numpy as np
# np.random.seed(1123)
# a1 = np.random.uniform(0, 0.4, size=[2,52,52,3,2]).astype(np.float32)
# np.random.seed(1233)
# a11 = np.random.uniform(0.5, 0.9, size=[2,52,52,3,2]).astype(np.float32)
# np.random.seed(123)
# a111 = np.random.uniform(size=[2,52,52,3,1]).astype(np.float32)
# a111 = a111+0.01>1.
#
# np.random.seed(1213)
# a1111 = np.random.uniform(size=[2,52,52,3,2]).astype(np.float32)
# a1 = np.concatenate([a1, a11, a111, a1111],axis=-1)
#
# np.random.seed(4156)
# a2 = np.random.uniform(size=[2,52,52,3,5+2]).astype(np.float32)
#
#
# # a3 = yolov3_loss(CFG,0)(a1,a2)
# # print(a3)
# # exit()
#
#
#
#
# # print(tf.boolean_mask(x1, x2))
# # exit()
# # a1 = np.ones(shape=[1,2,2,3,5+2]).astype(np.float32)
# # a2 = np.ones(shape=[1,2,2,3,5+2]).astype(np.float32)
# # o1 = Yolov3ObjectLoss.bce(CFG,0)(a1,a2)
#
# # print(o1)
# # exit()
# # regression_loss = Yolov3BoxRegressionLoss().mse(CFG,0)(a1,a2)
# # regression_loss = Yolov3BoxRegressionLoss().bce_and_l1loss(CFG,0)(a1,a2)
# regression_loss = Yolov3BoxRegressionLoss().ciou(CFG,0)(a1,a2)
# print(regression_loss)
# # print(CFG['anchors'])
# exit()
#
#
#
#
# np.random.seed(121)
# a1 = np.random.randint(low=0,high=2,size=[3,52,52,3,5+2]).astype(np.int32).astype(np.float32)
# np.random.seed(82)
# a2 = np.random.uniform(0, 1., size=[3,52,52,3,5+2]).astype(np.float32)
#
# classification_loss = Yolov3ClassificationLoss().focal_loss(CFG,0)(a1,a2)
# classification_loss = Yolov3ClassificationLoss.bce(CFG,0)(a1,a2)
# classification_loss = Yolov3ClassificationLoss.ce(CFG,0)(a1,a2)
# print(classification_loss)
# # print(a1)
# # print(a2)
# # print(a1*CFG['anchors'][0])
#
#
# #
# # np.random.seed(1123)
# # a1 = np.random.randint(low=0,high=2,size=[1,1,1,3,2]).astype(np.int32).astype(np.float32)
# #
# # np.random.seed(23)
# # a2 = np.random.uniform(0, 1., size=[1,1,1,3,2]).astype(np.float32)
# # classification_loss = Yolov3ClassificationLoss().focal_loss(CFG,0)(a1,a2)
# # # print(a1)
# # # print(a2)
# # print(classification_loss)
# #
# #
# # np.random.seed(121123)
# # a1 = np.random.randint(low=0,high=2,size=[1,1,1,3,2]).astype(np.int32).astype(np.float32)
# # np.random.seed(23123)
# # a2 = np.random.uniform(0, 1., size=[1,1,1,3,2]).astype(np.float32)
# # classification_loss = Yolov3ClassificationLoss().focal_loss(CFG,0)(a1,a2)
# # # print(a1)
# # # print(a2)
# # print(classification_loss)
#
#
#
# # print(CFG['anchors'])
# # #test
# # import numpy as np
# # a1 = np.array([[[1,2,3.0]],[[4,5,6]]])
# # a2 = np.array([[[1,2,3.0]],[[4,5,6]]])
# # print(a1[...,0:1])
# # out1 = tf.keras.losses.binary_crossentropy(a1[...,0:1],a1[...,0:1])
# # print(out1.shape)
# # print(out1)

