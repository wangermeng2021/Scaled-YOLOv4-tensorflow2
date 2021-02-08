

import tensorflow as tf

from model.CSPDarknet53 import scaled_yolov4_csp_darknet53
from model.head import head
from model.yolo_head import yolov3_head
from model.postprocess import postprocess
def Yolov4(args, training=True):
    input = tf.keras.layers.Input((None, None, 3))
    scaled_yolov4_csp_darknet53_outputs = scaled_yolov4_csp_darknet53(input,mode=args.model_type)
    head_outputs = head(scaled_yolov4_csp_darknet53_outputs)
    outputs = yolov3_head(head_outputs, args)

    if training:
        model = tf.keras.Model(inputs=input, outputs=outputs)
        return model
    else:
        # boxes, scores, classes, valid_detections = postprocess(outputs,params)
        # return tf.keras.Model(inputs=input, outputs=[boxes, scores, classes, valid_detections])
        pre_nms_decoded_boxes, pre_nms__scores = postprocess(outputs,args)
        return tf.keras.Model(inputs=input, outputs=[pre_nms_decoded_boxes, pre_nms__scores])

