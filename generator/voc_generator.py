
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
            sets = self._args.voc_val_set
        self.class_names_dict={}
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'dataset',self._args.class_names)) as f:
            self.class_names = f.read().splitlines()
            for i in range(len(self.class_names)):
                self.class_names_dict[self.class_names[i]] = i
        root_dir = self._args.dataset

        self.batch_size = self._args.batch_size
        self.skip_difficult = int(self._args.voc_skip_difficult)
        self.multi_scale = [int(item) for item in self._args.multi_scale.split(',')]
        self.augment = self._args.augment


        self.xml_path_list = []
        self.img_path_list = []
        sets_list = sets.split(',')
        sets_name = []
        for si in range(len(sets_list)//2):
            sets_name.append((sets_list[si * 2], sets_list[si * 2+1]))

        for voc_year, voc_set in sets_name:

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

        if mode==0:
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

            random_img_size = np.random.choice(self.multi_scale)
            self.max_side = self.min_side = random_img_size
            self.gluoncv_aug = aug_gluoncv.YOLO3DefaultTrainTransform(self.max_side, self.max_side)
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
            tail_batch_size = len(batch_boxes_list)
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

            batch_img = batch_img[0:tail_batch_size]
            batch_boxes = batch_boxes[0:tail_batch_size]
            groundtruth_valids = groundtruth_valids[0:tail_batch_size]

            batch_img, batch_boxes = preprocess(batch_img, batch_boxes)
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
