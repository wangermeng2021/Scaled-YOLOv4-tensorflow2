
import tensorflow as tf
from model.common import conv2d_bn_mish

def yolov3_optimizers(args):
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(momentum=args.momentum, nesterov=args.nesterov, name='sgd')
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(name='adam')
    else:
        raise ValueError("{} is invalid!".format(args.optimizer))
    return optimizer

