import math
import tensorflow as tf
from model.common import conv2d_bn_mish
from  model.common import scaled_yolov4_csp_block

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def csp_darknet_block(x, loop_num, filters, is_half_filters=True):

    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_mish(x, filters, (3,3), strides=(2,2), padding='valid')
    csp_branch = conv2d_bn_mish(x, filters//2 if is_half_filters else filters, (1, 1))
    darknet_branch = conv2d_bn_mish(x, filters//2 if is_half_filters else filters, (1, 1))

    for i in range(loop_num):
        x = conv2d_bn_mish(darknet_branch, filters//2, (1, 1))
        x = conv2d_bn_mish(x, filters // 2 if is_half_filters else filters, (3, 3))
        darknet_branch = tf.keras.layers.Add()([darknet_branch, x])

    darknet_branch = conv2d_bn_mish(darknet_branch, filters // 2 if is_half_filters else filters, (1, 1))

    x = tf.keras.layers.Concatenate()([darknet_branch, csp_branch])

    return conv2d_bn_mish(x, filters, (1, 1))

def scaled_yolov4_csp_darknet53(x,mode='p5'):
    darknet53_filters = [64 * 2 ** i for i in range(5)]
    no = (8 // 2) * (80 + 5) # (len(anchor)//2) * (nc+5)
    gw = 1.25 #width multiplier
    c2 = 32
    
    if mode == 'p5':
        loop_nums = [1, 3, 15, 15, 7]
    elif mode == 'p6':
        loop_nums = [1, 3, 15, 15, 7, 7]
        darknet53_filters += [1024]
    elif mode == 'p7':
        loop_nums = [1, 3, 15, 15, 7, 7, 7]
        darknet53_filters += [1024]*2
        c2 = make_divisible(c2 * gw , 8) if c2 != no else c2
 

    x = conv2d_bn_mish(x, c2, (3, 3), name="first_block")
    output_layers = []

    for block_index in range(len(loop_nums)):
        
        if mode == 'p7':
            c2 = darknet53_filters[block_index]
            c2 = make_divisible(c2 * gw , 8) if c2 != no else c2
            x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            x = conv2d_bn_mish(x,c2 , (3, 3), strides=(2, 2), padding='valid',name="backbone_block_{}_0".format(block_index))
            x = scaled_yolov4_csp_block(x, c2,loop_nums[block_index], type="backbone",name="backbone_block_{}_1".format(block_index))
            output_layers.append(x)  
        
        else:
            x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            x = conv2d_bn_mish(x, darknet53_filters[block_index], (3, 3), strides=(2, 2), padding='valid',name="backbone_block_{}_0".format(block_index))
            x = scaled_yolov4_csp_block(x, darknet53_filters[block_index],loop_nums[block_index], type="backbone",name="backbone_block_{}_1".format(block_index))
            output_layers.append(x)
    
    return output_layers[2:]



