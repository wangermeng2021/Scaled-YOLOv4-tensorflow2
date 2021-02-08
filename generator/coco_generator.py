
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

        self.batch_size = self._args.batch_size
        self.skip_difficult = int(self._args.voc_skip_difficult)
        self.multi_scale =  [int(item) for item in self._args.multi_scale.split(',')]
        self.augment = self._args.augment


        self.coco      = COCO(os.path.join(self._args.dataset, 'annotations', 'instances_' + self.sets + '.json'))
        self.image_ids = self.coco.getImgIds()
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels_inverse[c['id']] = len(self.coco_labels_inverse)
        self.img_path_list = []
        self.boxes_and_labels = []
        for id in self.image_ids:
            image_info = self.coco.imgs[id]
            self.img_path_list.append(os.path.join(self._args.dataset, 'images', self.sets, image_info['file_name']))

        for image_index in self.image_ids:
            self.boxes_and_labels.append(self.parse_json(image_index))
        if mode == 0:
            pad_num = 4*self.batch_size-len(self.img_path_list) % (4*self.batch_size)
            for _ in range(pad_num):
                pi = np.random.choice(range(len(self.img_path_list)))
                self.img_path_list.append(self.img_path_list[pi])
                self.boxes_and_labels.append(copy.deepcopy(self.boxes_and_labels[pi]))

        self.data_index = np.empty([len(self.img_path_list)], np.int32)
        for index in range(len(self.img_path_list)):
            self.data_index[index] = index

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
        return int(self._args.num_classes)
    def get_size(self):
        return len(self.img_path_list)
    def __len__(self):
        if self.mode == 0:
            return len(self.img_path_list) // self.batch_size
        else:
            return int(np.ceil(len(self.img_path_list) / self.batch_size))
    def on_epoch_end(self):
        if self.mode == 0:
            np.random.shuffle(self.data_index)
    def __getitem__(self, item):
        with tf.device("/cpu:0"):
            groundtruth_valids = np.zeros([self.batch_size],np.int)

            # if self.multi_scale:
            random_img_size = np.random.choice(self.multi_scale)
            self.max_side = self.min_side = random_img_size
            self.gluoncv_aug = aug_gluoncv.YOLO3DefaultTrainTransform(self.max_side, self.min_side)
            batch_img = np.zeros([self.batch_size, self.max_side, self.max_side, 3])
            batch_boxes = np.empty([self.batch_size, self._args.max_box_num_per_image, 5])
            batch_boxes_list = []
            # start_time = datetime.datetime.now()
            # iii = 0
            for batch_index, file_index in enumerate(self.data_index[item*self.batch_size:(item+1)*self.batch_size]):
                #get image from file
                # image_info = self.coco.loadImgs(self.image_ids[file_index])[0]
                # image_info = self.coco.imgs[self.image_ids[file_index]]
                # img_path = os.path.join(self._args.dataset, 'images', self.sets, image_info['file_name'])
                img_path = self.img_path_list[file_index]
                img = self.read_img(img_path)
                img, scale, pad = self.resize_fun(img, (self.max_side, self.min_side))
                batch_img[batch_index, 0:img.shape[0], 0:img.shape[1], :] = img
                boxes = self.boxes_and_labels[file_index]
                boxes = copy.deepcopy(boxes)
                # print(iii)
                # iii+=1
                boxes[:, 0:4] *= scale
                half_pad = pad // 2
                boxes[:, 0:4] += np.tile(half_pad, 2)
                batch_boxes_list.append(boxes)
                groundtruth_valids[batch_index] = boxes.shape[0]
                boxes = np.pad(boxes, [(0, self._args.max_box_num_per_image-boxes.shape[0]), (0, 0)], mode='constant')
                batch_boxes[batch_index] = boxes
            # print("hhhhhhhhhhh")
            tail_batch_size = len(batch_boxes_list)
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

            batch_img = batch_img[0:tail_batch_size]
            batch_boxes = batch_boxes[0:tail_batch_size]
            groundtruth_valids = groundtruth_valids[0:tail_batch_size]
            ###############
            batch_img, batch_boxes = preprocess(batch_img, batch_boxes)
            ###############
            if int(self._args.num_classes) == 1:
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

