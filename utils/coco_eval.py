
import numpy as np
from utils import  coco_tools
class CocoEvalidation():
    def __init__(self, groundtruth_boxes,groundtruth_classes,groundtruth_valids,class_names):
        self.groundtruth_boxes = groundtruth_boxes
        self.groundtruth_classes = groundtruth_classes
        self.groundtruth_valids = groundtruth_valids
        self.class_names = class_names

        self.groundtruth_dict = self.convert_gt_to_coco(groundtruth_boxes,groundtruth_classes,groundtruth_valids,class_names)
        # print(self.groundtruth_dict)
        self.groundtruth = coco_tools.COCOWrapper(self.groundtruth_dict)

        pass
    def convert_gt_to_coco(self, groundtruth_boxes,groundtruth_classes,groundtruth_valids,class_names):

        categories=[{'id': id,'name': name} for id, name in  enumerate(class_names)]
        annotation_id = 1
        num_imgs = groundtruth_classes.shape[0]
        if num_imgs == 0:
            raise ValueError('the number of groundtruth_boxes must be greater than zero.')
        coco_groundtruth = []
        image_export_list = []
        for image_index in range(num_imgs):
          num_boxes = groundtruth_valids[image_index]
          for box_index in range(num_boxes):
            box_wh = groundtruth_boxes[image_index][box_index][2:4]-groundtruth_boxes[image_index][box_index][0:2]
            box_area = box_wh[0]*box_wh[1]
            export_dict = {
                'id':
                    annotation_id + box_index,
                'image_id':
                    image_index,
                'category_id':
                    int(groundtruth_classes[image_index][box_index]),
                'bbox':
                    list(np.concatenate([groundtruth_boxes[image_index][box_index][0:2],box_wh],axis=-1)),
                'area': box_area,
                'iscrowd': 0
            }
            coco_groundtruth.append(export_dict)
            image_export_list.append({'id': image_index})
          annotation_id += num_boxes

        groundtruth_dict = {
          'annotations': coco_groundtruth,
          'images': image_export_list,
          'categories': categories
        }
        return groundtruth_dict

    def convert_detection_to_coco(self, detection_boxes,detection_scores,detection_classes,detection_valids):

      num_images = detection_classes.shape[0]
      if detection_boxes.shape[0] == 0:
          raise ValueError('the number of detection_boxes must be greater than zero.')

      coco_groundtruth = []
      for img_index in range(num_images):
          num_boxes = detection_valids[img_index]
          for box_index in range(num_boxes):
              export_dict = {
                  'image_id':
                      img_index,
                  'category_id':
                      int(detection_classes[img_index,box_index]),
                  'bbox':
                      list(np.concatenate([detection_boxes[img_index, box_index][0:2], detection_boxes[img_index, box_index][2:4]-detection_boxes[img_index, box_index][0:2]], axis=-1)),
                  'score':
                      float(detection_scores[img_index,box_index]),
              }

              coco_groundtruth.append(export_dict)
      return coco_groundtruth


    def get_coco_mAP(self,detection_boxes,detection_scores,detection_classes,detection_valids):
        detections_list = self.convert_detection_to_coco(detection_boxes, detection_scores, detection_classes, detection_valids)
        # print(detections_list)
        detections = self.groundtruth.LoadAnnotations(detections_list)
        evaluator = coco_tools.COCOEvalWrapper(self.groundtruth, detections)
        summary_metrics, _ = evaluator.ComputeMetrics()
        return summary_metrics
#

