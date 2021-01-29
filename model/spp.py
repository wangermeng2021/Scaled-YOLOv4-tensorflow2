
import tensorflow as tf
from model.common import conv2d_bn_mish

def spp(x):
    pool_sizes = [5, 9, 13]
    pooling_results = [tf.keras.layers.MaxPooling2D((pool_size,pool_size), strides=(1, 1),padding='same')(x) for pool_size in pool_sizes]
    spp_result = tf.keras.layers.Concatenate()(pooling_results+[x])
    spp_result = conv2d_bn_mish(spp_result, x.shape[3], (1, 1))
    return spp_result

