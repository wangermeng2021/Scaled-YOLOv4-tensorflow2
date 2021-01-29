"""Train YOLOv3 with random shapes."""
import os
import time
import warnings
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# from data_augment import cutmix
from generator import data_augment
from generator.get_y_true import get_y_true,get_y_true_with_one_class
from utils.preprocess import preprocess
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from utils import aug_gluoncv
import copy
from utils.preprocess import resize_img,resize_img_aug
class VocGenerator(tf.keras.utils.Sequence):

    def __init__(self, args, mode=0):
        self._args = args
        self.mode = mode
        if self.mode == 0:
            sets = self._args.voc_train_set
        else:
            sets = self._args.voc_valid_set
        self.class_names_dict={}
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'dataset',self._args.class_names)) as f:
            self.class_names = f.read().splitlines()
            for i in range(len(self.class_names)):
                self.class_names_dict[self.class_names[i]] = i
        root_dir = self._args.dataset

        self.batch_size = self._args.batch_size
        self.skip_difficult = int(self._args.voc_skip_difficult)
        # self.multi_scale = self._args.multi_scale
        self.multi_scale = [int(item) for item in self._args.multi_scale.split(',')]
        self.augment = self._args.augment

        self.gluoncv_aug = aug_gluoncv.YOLO3DefaultTrainTransform(self.multi_scale[0], self.multi_scale[0])

        self.xml_path_list = []
        self.img_path_list = []
        for voc_year, voc_set in sets:
            txt_path = os.path.join(root_dir, str(voc_year), 'ImageSets', 'Main', voc_set+'.txt')
            with open(txt_path) as f:
                lines = f.readlines()
            for line in lines:
                valid_label = self.check_img_and_xml(os.path.join(root_dir,  str(voc_year), 'JPEGImages', line.strip() + '.jpg'),os.path.join(root_dir, str(voc_year), 'Annotations', line.strip()+'.xml'))
                if valid_label:
                    self.xml_path_list.append(os.path.join(root_dir, str(voc_year), 'Annotations', line.strip()+'.xml'))
                    self.img_path_list.append(os.path.join(root_dir, str(voc_year), 'JPEGImages', line.strip() + '.jpg'))

        self.boxes_and_labels = []
        for xml_path in self.xml_path_list:
            self.boxes_and_labels.append(self.parse_xml(xml_path))
        self.data_index = np.empty([len(self.xml_path_list)], np.int32)
        for index in range(len(self.xml_path_list)):
            self.data_index[index] = index
        if self.mode == 0:
            self.resize_fun = resize_img_aug
            np.random.shuffle(self.data_index)
            if self.augment == 'mosaic':
                self.batch_size *= 4
        else:
            self.resize_fun = resize_img
            self.augment = None

    def check_img_and_xml(self, img_path, xml_path):
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
            num_valid_boxes = 0
            for element in xml_root.iter('object'):
                # truncated = int(element.find('truncated').text)
                difficult = int(element.find('difficult').text)
                if difficult:
                    continue
                num_valid_boxes += 1
            if num_valid_boxes == 0:
                return False
        except:
            return False
        return True

    def parse_xml(self,file_path):
        try:
            tree = ET.parse(file_path)
            xml_root = tree.getroot()

            size = xml_root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            boxes = np.empty((len(xml_root.findall('object')), 5))
            box_index = 0
            for i, element in enumerate(xml_root.iter('object')):
                # truncated = int(element.find('truncated').text)
                difficult = int(element.find('difficult').text)
                class_name = element.find('name').text
                box = np.zeros((4,))
                label = self.class_names_dict[class_name]
                bndbox = element.find('bndbox')

                box[0] = float(bndbox.find('xmin').text)-1
                box[1] = float(bndbox.find('ymin').text)-1
                box[2] = float(bndbox.find('xmax').text)-1
                box[3] = float(bndbox.find('ymax').text)-1

                # assert 0 <= box[0] < width
                # assert 0 <= box[1] < height
                # assert box[0] < box[2] < width
                # assert box[1] < box[3] < height
                box[0] = np.maximum(box[0], 0)
                box[1] = np.maximum(box[1], 0)
                box[2] = np.minimum(box[2], width-1)
                box[3] = np.minimum(box[3], height-1)

                # if truncated and self.skip_truncated:
                #     continue
                if difficult and self.skip_difficult:
                    continue

                boxes[box_index, 0:4] = box
                boxes[box_index, 4] = int(label)
                box_index += 1
            return boxes[0:box_index]
            # return boxes[boxes[...,3]>0]
        except ET.ParseError as e:
            ValueError('there is an error in parsing xml file: {}: {}'.format(file_path, e))
    def get_classes_num(self):
        return self._args.num_classes
    def get_size(self):
        return len(self.img_path_list)
    def __len__(self):
        return len(self.xml_path_list) // self.batch_size
    def on_epoch_end(self):
        if self.mode == 0:
            np.random.shuffle(self.data_index)
    def __getitem__(self, item):

        groundtruth_valids = np.zeros([self.batch_size],np.int)

        random_img_size = np.random.choice(self.multi_scale)
        self.max_side = self.min_side = random_img_size

        batch_img = np.zeros([self.batch_size, self.max_side, self.max_side, 3])
        batch_boxes = np.empty([self.batch_size, self._args.max_box_num_per_image, 5])
        batch_boxes_list = []

        for batch_index, file_index in enumerate(self.data_index[item*self.batch_size:(item+1)*self.batch_size]):
            #get image from file
            img_path = self.img_path_list[file_index]
            img = self.read_img(img_path)

            img, scale, pad = self.resize_fun(img, (self.max_side, self.min_side))
            batch_img[batch_index, 0:img.shape[0], 0:img.shape[1], :] = img
            boxes = self.boxes_and_labels[file_index]
            boxes = copy.deepcopy(boxes)
            boxes[:, 0:4] *= scale
            half_pad = pad // 2
            boxes[:, 0:4] += np.tile(half_pad,2)
            batch_boxes_list.append(boxes)
            groundtruth_valids[batch_index] = boxes.shape[0]
            boxes = np.pad(boxes, [(0, self._args.max_box_num_per_image-boxes.shape[0]), (0, 0)], mode='constant')
            batch_boxes[batch_index] = boxes

        #augment
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

        batch_img, batch_boxes = preprocess(batch_img, batch_boxes)

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
# #
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
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
#     parser.add_argument('--dataset', default='../dataset/raccoon_voc')
#     parser.add_argument('--dataset-type', default='voc',help="voc,coco")
#     parser.add_argument('--voc-train-set', default=[('dataset_1', 'train')],help="VOC dataset:[(VOC2007, 'trainval'), (VOC2012, 'trainval')]")
#     parser.add_argument('--voc-valid-set', default=[('dataset_1', 'val')],help="VOC dataset:[(VOC2007, 'test')]")
#     parser.add_argument('--voc-skip-difficult', default=True)
#     parser.add_argument('--coco-train-set', default='train2017')
#     parser.add_argument('--coco-valid-set', default='val2017')
#     parser.add_argument('--num-classes', default=1)
#     parser.add_argument('--class-names', default='raccoon.names')
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
#     voc_generator = VocGenerator(args, mode=1)
#
#     for imgs, y_true,boxes in voc_generator:
#         for box1_index, box1 in enumerate(boxes):
#             for box2 in box1:
#                 if box2[2] == 0:
#                     continue
#                 box = box2[0:4] * np.tile(imgs[box1_index].shape[0:2][::-1], [2])
#                 box = box.astype(np.int32)
#                 cv2.rectangle(imgs[box1_index],(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
#             cv2.imshow('1', imgs[box1_index])
#             cv2.waitKey(0)


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
from model.yolov4 import Yolov4
#
#
#
# if __name__=="__main__":
#     voc_generator = VocGenerator(args, mode=0)
#     # print("bbbbbbbbbbbb")
#     # model = Yolov4(CFG, training=False)
#     # model = tf.keras.models.load_model('/home/wangem1/yolov4/yolov4/pretrained/yolov4-sam-mish.tf')
#     for imgs, y_true,boxes in voc_generator:
#
#         # boxes,scores,classes,valid_detections = model.predict(imgs)
#         #
#         # draw_boxes_on_batch_img(imgs,boxes,scores,classes,valid_detections,class_names)
#         # continue
# #
#         for box1_index, box1 in enumerate(boxes):
#
#             for box2 in box1:
#                 if box2[2]==0:
#                     continue
#                 print(box2)
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
# # #
