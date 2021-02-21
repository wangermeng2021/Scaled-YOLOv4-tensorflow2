
import tensorflow as tf
import os
import numpy as np

from model.postprocess import postprocess
from tensorflow.keras import callbacks
from utils.coco_eval import CocoEvalidation
from model.nms import yolov4_nms
from tqdm import tqdm
class CocoMapCallback(callbacks.Callback):

    def __init__(self, pred_generator,model,args,mAP_writer):
        self.args = args
        self.pred_generator = pred_generator
        self.model = model

        self.mAP_writer = mAP_writer
        self.max_coco_map = -1
        self.max_coco_map_epoch = -1
        self.best_weight_path = ''

        groundtruth_boxes = []
        groundtruth_classes = []
        groundtruth_valids = []
        print("loading dataset...")
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),os.path.join('dataset',args.class_names))) as f:
            class_names = [name.strip() for name in f.readlines()]
        pred_generator_tqdm = tqdm(self.pred_generator, total=len(self.pred_generator))
        for batch_img, batch_boxes, batch_valids in pred_generator_tqdm:
            groundtruth_boxes.append(batch_boxes[..., 0:4])
            groundtruth_classes.append(batch_boxes[..., 4])
            groundtruth_valids.append(batch_valids)
        groundtruth_boxes = np.concatenate(groundtruth_boxes, axis=0)
        groundtruth_classes = np.concatenate(groundtruth_classes, axis=0)
        groundtruth_valids = np.concatenate(groundtruth_valids, axis=0)

        self.coco = CocoEvalidation(groundtruth_boxes,groundtruth_classes,groundtruth_valids,class_names)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.args.start_eval_epoch or epoch % self.args.eval_epoch_interval != 0:
            return

        detection_boxes = []
        detection_scores = []
        detection_classes = []
        detection_valids = []

        pred_generator_tqdm = tqdm(self.pred_generator, total=len(self.pred_generator))
        for batch_img, _, _ in pred_generator_tqdm:
            model_outputs = self.model.predict(batch_img)
            pre_nms_decoded_boxes, pre_nms_scores = postprocess(model_outputs, self.args)
            pre_nms_decoded_boxes = pre_nms_decoded_boxes.numpy()
            pre_nms_scores = pre_nms_scores.numpy()

            boxes, scores, classes, valid_detections = yolov4_nms(self.args)(pre_nms_decoded_boxes, pre_nms_scores,
                                                                             self.args)

            detection_boxes.append(boxes)
            detection_scores.append(scores)
            detection_classes.append(classes)
            detection_valids.append(valid_detections)
            pred_generator_tqdm.set_description("Evaluation...")
        detection_boxes = np.concatenate(detection_boxes, axis=0)
        detection_scores = np.concatenate(detection_scores, axis=0)
        detection_classes = np.concatenate(detection_classes, axis=0)
        detection_valids = np.concatenate(detection_valids, axis=0)

        summary_metrics = self.coco.get_coco_mAP(detection_boxes, detection_scores, detection_classes, detection_valids)

        if summary_metrics['Precision/mAP@.50IOU'] > self.max_coco_map:
            self.max_coco_map = summary_metrics['Precision/mAP@.50IOU']
            self.max_coco_map_epoch = epoch
            self.best_weight_path = os.path.join(self.args.checkpoints_dir, 'best_weight_{}_{}_{:.3f}'.format(self.args.model_type,self.max_coco_map_epoch, self.max_coco_map))
            self.model.save_weights(self.best_weight_path)

        print("max_coco_map:{},epoch:{}".format(self.max_coco_map, self.max_coco_map_epoch))
        with self.mAP_writer.as_default():
            tf.summary.scalar("mAP@0.5", summary_metrics['Precision/mAP@.50IOU'], step=epoch)
            self.mAP_writer.flush()