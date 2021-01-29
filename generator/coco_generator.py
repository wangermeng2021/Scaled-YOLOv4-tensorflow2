"""Train YOLOv3 with random shapes."""
import os

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# from data_augment import cutmix
from generator import data_augment
from generator.get_y_true import get_y_true,get_y_true_with_one_class
from utils.preprocess import preprocess

# from data_augment import np_random_color_distort
# from utils import  aug_gluoncv

from pycocotools.coco import COCO

from utils import aug_gluoncv
import copy
from utils.preprocess import resize_img,resize_img_aug
class CocoGenerator(tf.keras.utils.Sequence):

    def __init__(self, args, mode=0):
        self._args = args
        self.mode = mode
        if self.mode == 0:
            self.sets = self._args.coco_train_set
        else:
            self.sets = self._args.coco_valid_set

        self.coco      = COCO(os.path.join(self._args.dataset, 'annotations', 'instances_' + self.sets + '.json'))
        self.image_ids = self.coco.getImgIds()
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels_inverse[c['id']] = len(self.coco_labels_inverse)
        self.boxes_and_labels = []
        for image_index in self.image_ids:
            self.boxes_and_labels.append(self.parse_json(image_index))
        self.data_index = np.empty([len(self.image_ids)], np.int32)
        for index in range(len(self.image_ids)):
            self.data_index[index] = index

        self.batch_size = self._args.batch_size
        self.skip_difficult = int(self._args.voc_skip_difficult)
        # self.multi_scale = self._args.multi_scale
        self.multi_scale =  [int(item) for item in self._args.multi_scale.split(',')]
        self.augment = self._args.augment

        self.gluoncv_aug = aug_gluoncv.YOLO3DefaultTrainTransform(self.multi_scale[0], self.multi_scale[0])

        if self.mode == 0:
            self.resize_fun = resize_img_aug
            np.random.shuffle(self.data_index)
            if self.augment == 'mosaic':
                self.batch_size *= 4
        else:
            self.resize_fun = resize_img
            self.augment = None

    def parse_json(self,image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_index, iscrowd=False)
        labels = np.empty((0,))
        boxes = np.empty((0, 4))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return np.empty((0, 5))
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            labels = np.concatenate([labels, [self.coco_labels_inverse[a['category_id']]]], axis=0)
            boxes = np.concatenate([boxes, [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
        labels = np.expand_dims(labels, axis=-1)
        return np.concatenate([boxes, labels], axis=-1)
    def get_classes_num(self):
        return self._args.num_classes
    def get_size(self):
        return len(self.image_ids)
        # return len(self.img_path_list)
    def __len__(self):
        return len(self.image_ids)// self.batch_size
    def on_epoch_end(self):
        if self.mode == 0:
            np.random.shuffle(self.data_index)
    def __getitem__(self, item):
        groundtruth_valids = np.zeros([self.batch_size],np.int)

        # if self.multi_scale:
        random_img_size = np.random.choice(self.multi_scale)
        self.max_side = self.min_side = random_img_size

        batch_img = np.zeros([self.batch_size, self.max_side, self.max_side, 3])
        batch_boxes = np.empty([self.batch_size, self._args.max_box_num_per_image, 5])
        batch_boxes_list = []
        for batch_index, file_index in enumerate(self.data_index[item*self.batch_size:(item+1)*self.batch_size]):
            #get image from file
            image_info = self.coco.loadImgs(self.image_ids[file_index])[0]
            img_path = os.path.join(self._args.dataset, 'images', self.sets, image_info['file_name'])
            img = self.read_img(img_path)
            img, scale, pad = self.resize_fun(img, (self.max_side, self.min_side))
            batch_img[batch_index, 0:img.shape[0], 0:img.shape[1], :] = img
            boxes = self.boxes_and_labels[file_index]
            boxes = copy.deepcopy(boxes)

            boxes[:, 0:4] *= scale
            half_pad = pad // 2
            boxes[:, 0:4] += np.tile(half_pad, 2)
            batch_boxes_list.append(boxes)
            groundtruth_valids[batch_index] = boxes.shape[0]
            boxes = np.pad(boxes, [(0, self._args.max_box_num_per_image-boxes.shape[0]), (0, 0)], mode='constant')
            batch_boxes[batch_index] = boxes

        ###############
        #augment
        # self.mosaic = False
        if self.augment == 'mosaic':
            new_batch_size = self.batch_size//4
            for bi in range(new_batch_size):
                four_img, four_boxes, one_img, one_boxes = data_augment.load_mosaic(batch_img[bi * 4:(bi + 1) * 4],
                                                                                    batch_boxes_list[bi * 4:(bi + 1) * 4])
                data_augment.random_hsv(one_img)
                data_augment.random_left_right_flip(one_img, one_boxes)
                groundtruth_valids[bi] = one_boxes.shape[0]
                one_boxes = np.pad(one_boxes,[(0, self._args.max_box_num_per_image-one_boxes.shape[0]), (0, 0)], mode='constant')
                batch_img[bi] = one_img
                batch_boxes[bi] = one_boxes

            batch_img = batch_img[0:new_batch_size]
            batch_boxes = batch_boxes[0:new_batch_size]
        elif self.augment == 'only_flip_left_right':
            for bi in range(self.batch_size):
                data_augment.random_left_right_flip(batch_img[bi], batch_boxes[bi])
        elif self.augment == 'ssd_random_crop':
            batch_img = batch_img.astype(np.uint8)
            for di in range(self.batch_size):
                batch_img[di], batch_boxes_list[di] = self.gluoncv_aug(batch_img[di], batch_boxes_list[di])
                batch_boxes[di] = np.pad(batch_boxes_list[di], [(0, self._args.max_box_num_per_image - batch_boxes_list[di].shape[0]), (0, 0)])
                groundtruth_valids[di] = batch_boxes_list[di].shape[0]

        ###############
        batch_img, batch_boxes = preprocess(batch_img, batch_boxes)
        ###############
        if self._args.num_classes == 1:
            y_true = get_y_true_with_one_class(self.max_side, batch_boxes, groundtruth_valids, self._args)
        else:
            y_true = get_y_true(self.max_side, batch_boxes, groundtruth_valids, self._args)

        if self.mode == 2:
            return batch_img, batch_boxes, groundtruth_valids
        return batch_img, y_true
        return batch_img, y_true, batch_boxes

    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1]



#
#
# #
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# def parse_args(args):
#     parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
#     #training
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--batch-size', default=4, type=int)
#     parser.add_argument('--start-eval-epoch', default=50, type=int)
#     parser.add_argument('--eval-epoch-interval', default=1)
#     #model
#     parser.add_argument('--model-type', default='p5', help="choices=['p5','p6','p7']")
#     parser.add_argument('--pretrained-weights', default='pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain',help="Path to a pretrain weights.")
#     parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
#     #loss
#     parser.add_argument('--box-regression-loss', default='diou',help="choices=['giou','diou','ciou']")
#     parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
#     parser.add_argument('--focal-alpha', default= 0.25)
#     parser.add_argument('--focal-gamma', default=2.0)
#     parser.add_argument('--ignore-thr', default=0.7)
#     parser.add_argument('--reg-losss-weight', default=0.05)
#     parser.add_argument('--obj-losss-weight', default=1.0)
#     parser.add_argument('--cls-losss-weight', default=0.5)
#     #dataset
#     parser.add_argument('--dataset', default='/media/wangem1/wem/dataset/coco/annotations_trainval2017')
#     parser.add_argument('--dataset-type', default='coco',help="voc,coco")
#     parser.add_argument('--voc-train-set', default=[('dataset_1', 'train')],help="VOC dataset:[(VOC2007, 'trainval'), (VOC2012, 'trainval')]")
#     parser.add_argument('--voc-valid-set', default=[('dataset_1', 'val')],help="VOC dataset:[(VOC2007, 'test')]")
#     parser.add_argument('--voc-skip-difficult', default=True)
#     parser.add_argument('--coco-train-set', default='train2017')
#     parser.add_argument('--coco-valid-set', default='val2017')
#     parser.add_argument('--num-classes', default=80)
#     parser.add_argument('--class-names', default='coco.names')
#     parser.add_argument('--augment', default='ssd_random_crop',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
#     parser.add_argument('--multi-scale', default='352',help="Input data shapes for training, use 320+32*i(i>=0)")#896
#     parser.add_argument('--max-box-num-per-image', default=100)
#     #optimizer
#     parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
#     parser.add_argument('--momentum', default=0.9)
#     parser.add_argument('--nesterov', default=True)
#     parser.add_argument('--weight-decay', default=5e-4)
#     #lr scheduler
#     parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
#     parser.add_argument('--init-lr', default=1e-3)
#     parser.add_argument('--lr-decay', default=0.1)
#     parser.add_argument('--lr-decay-epoch', default=[160, 180], type=int)
#     parser.add_argument('--warmup-epochs', default=0)
#     parser.add_argument('--warmup-lr', default=1e-4)
#     #postprocess
#     parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
#     parser.add_argument('--nms-max-box-num', default=300)
#     parser.add_argument('--nms-iou-threshold', default=0.2)
#     parser.add_argument('--score-threshold', default=0.5)
#     #anchor
#     parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
#     parser.add_argument('--anchor-match-iou_thr', default=0.2)
#     parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0)
#
#     parser.add_argument('--label-smooth', default=0.0)
#     parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
#     parser.add_argument('--accumulated-gradient-num', default=1)
#
#     return parser.parse_args(args)
# import argparse
# import sys
#
#
#
# if __name__ == "__main__":
#     args = parse_args(sys.argv[1:])
#     coco_generator = CocoGenerator(args, mode=0)
#
#     with open('/home/wangem1/papers_coding/Scaled-YOLOv4-tensorflow2/dataset/coco.names') as f:
#         class_names = f.read().splitlines()
#     for imgs, y_true,boxes in coco_generator:
#         for box1_index, box1 in enumerate(boxes):
#             print("fffffffffffffffff")
#             for box2 in box1:
#                 if box2[2] == 0:
#                     continue
#                 box = box2[0:4] * np.tile(imgs[box1_index].shape[0:2][::-1], [2])
#                 box = box.astype(np.int32)
#
#                 print(box2[4],class_names[int(box2[4])])
#                 cv2.rectangle(imgs[box1_index],(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
#             cv2.imshow('1', imgs[box1_index])
#             cv2.waitKey(0)
#
#
#



#
# #
# # ###test###
#
# # #
# #
# #
# #
# #
# # with open('/home/wangem1/yolov4/yolov4/dataset/voc.names') as f:
# #     class_names = f.read().splitlines()
# #
# # def draw_boxes_on_batch_img(img,boxes,scores,classes,valid_detections,class_names):
# #     for i in range(len(boxes)):
# #         for j in range(valid_detections[i]):
# #             if scores[i][j]<0.5:
# #                 continue
# #             x1y1 = (boxes[i][j][0:2] * img[i].shape[0:2][::-1]).astype(np.int)
# #             x2y2 = (boxes[i][j][2:4] * img[i].shape[0:2][::-1]).astype(np.int)
# #             cv2.rectangle(img[i], tuple(x1y1), tuple(x2y2), (0, 255, 0), 2)
# #             cv2.putText(img[i], str(class_names[int(classes[i][j])]) + " " + str("%0.2f" % scores[i][j]),
# #                         (x1y1[0], max(x1y1[1] - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# #         cv2.imshow("%d"%i, img[i])
# #         cv2.waitKey(0)
# #
# from model.yolov4 import Yolov4
# from config.param_config import CFG
#
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
# if __name__=="__main__":
#     voc_generator = CocoGenerator(CFG, mode=0)
#     # print("bbbbbbbbbbbb")
#     # model = Yolov4(CFG, training=False)
#     # model = tf.keras.models.load_model('/home/wangem1/yolov4/yolov4/pretrained/yolov4-sam-mish.tf')
#     for imgs, y_true, boxes in voc_generator:
#
#         # boxes,scores,classes,valid_detections = model.predict(imgs)
#         #
#         # draw_boxes_on_batch_img(imgs,boxes,scores,classes,valid_detections,class_names)
#         # continue
#
#         for box1_index, box1 in enumerate(boxes):
#
#             for box2 in box1:
#                 if box2[2]==0:
#                     continue
#                 # print(box2)
#
#                 # print("ssssssssssss")
#                 box = box2[0:4] * np.tile(imgs[box1_index].shape[0:2][::-1], [2])
#                 box = box.astype(np.int32)
#                 cv2.rectangle(imgs[box1_index],(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
#             cv2.imshow('1', imgs[box1_index])
#             cv2.waitKey(0)
# #         #
# #         # grid_h ,grid_w= y_true.shape[2],y_true.shape[3]
# #         #
# #         # for i in range(3):
# #         #     for h in range(grid_h):
# #         #         for w in range(grid_w):
# #         #             for i1 in range(3):
# #         #                 if y_true[0][i][h][w][i1][4]==1:
# #         #                     print(i,h,w,i1)
# #         #                     box = y_true[0][i][h][w][i1][0:4] * np.tile(imgs[0].shape[0:2][::-1], [2])
# #         #                     box = box.astype(np.int32)
# #         #                     cv2.rectangle(imgs[0], (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
# #         #                     cv2.imshow('123',imgs[0]/255)
# #         #                     cv2.waitKey(0)
# #
# #         # cv2.imshow('1',imgs[0])
# #         # cv2.imshow('2', imgs[1])
# #         # cv2.imshow('3', imgs[2])
# #         # cv2.imshow('4', imgs[3])
# #         # cv2.waitKey(0)

