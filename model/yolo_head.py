
import tensorflow as tf
from model.common import conv2d_bn_mish
from utils.anchors import yolo_anchors
def yolov3_head(inputs, args):

    class_num = int(args.num_classes)
    if class_num == 1:
        class_num = 0
    anchors = yolo_anchors[args.model_type]
    output_layers = []
    for index, x in enumerate(inputs):
        x = conv2d_bn_mish(x, x.shape[-1]*2, (3, 3), name='yolov3_head_%d_0'%index)
        x = tf.keras.layers.Conv2D(len(anchors[index])*(class_num+5), (1, 1), use_bias=True, name='yolov3_head_%d_1_conv2d'%index)(x)
        x = tf.reshape(x,[tf.shape(x)[0], tf.shape(x)[1],tf.shape(x)[2],-1,class_num+5])
        output_layers.append(x)

    return output_layers



