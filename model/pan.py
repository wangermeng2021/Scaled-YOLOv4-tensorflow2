

import tensorflow as tf
from model.common import conv2d_bn_mish
from model.common import scaled_yolov4_csp_block

def pan_down(backbone_output_layers):
    output_layers = []
    pan_down_output_layer = scaled_yolov4_csp_block(backbone_output_layers[-1],backbone_output_layers[-1].shape[-1]//2,3,type='spp')
    output_layers.append(pan_down_output_layer)
    for layer_index in range(2,len(backbone_output_layers)+1):
        pan_down_output_layer = conv2d_bn_mish(pan_down_output_layer, pan_down_output_layer.shape[-1] // 2,(1, 1))
        pan_down_output_layer = tf.keras.layers.UpSampling2D(size=(2, 2))(pan_down_output_layer)

        backbone_output_layer = conv2d_bn_mish(backbone_output_layers[-layer_index], backbone_output_layers[-layer_index].shape[3] // 2,(1, 1))
        pan_down_output_layer = tf.keras.layers.Concatenate()([backbone_output_layer, pan_down_output_layer])
        pan_down_output_layer = scaled_yolov4_csp_block(pan_down_output_layer,pan_down_output_layer.shape[-1] // 2, 3, type='head')
        output_layers.append(pan_down_output_layer)
    return output_layers

def pan_up(pan_down_output_layers):
    output_layers = []
    pan_up_output_layers = pan_down_output_layers[-1]
    output_layers.append(pan_up_output_layers)
    for layer_index in range(2, len(pan_down_output_layers) + 1):
        pan_up_output_layers = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(pan_up_output_layers)
        pan_up_output_layers = conv2d_bn_mish(pan_up_output_layers, pan_up_output_layers.shape[-1]*2, (3,3), strides=(2,2), padding='valid')
        pan_up_output_layers = tf.keras.layers.Concatenate()([pan_up_output_layers, pan_down_output_layers[-layer_index]])
        pan_down_output_layer = scaled_yolov4_csp_block(pan_up_output_layers, pan_up_output_layers.shape[-1] // 2, 3, type='head')
        output_layers.append(pan_down_output_layer)
    return output_layers

def pan(x):
    inputs = []
    for i in range(len(x)):
        inputs.append(tf.keras.layers.Input((None, None, x[i].shape[-1])))
    x = inputs
    x = pan_down(x)
    outputs = pan_up(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='pan')
