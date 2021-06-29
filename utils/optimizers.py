
import tensorflow as tf
from model.common import conv2d_bn_mish
from utils.sam import SAMOptimizer
def yolov3_optimizers(args):
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr,momentum=args.momentum, nesterov=args.nesterov, name='sgd')
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr,name='adam')
    elif args.optimizer == "SAM_sgd":
        return SAMOptimizer(tf.keras.optimizers.SGD(learning_rate=args.init_lr,momentum=args.momentum, nesterov=args.nesterov, name='sgd'))
    elif args.optimizer == "SAM_adam":
        return SAMOptimizer(tf.keras.optimizers.Adam(learning_rate=args.init_lr,name='adam'))
    else:
        raise ValueError("{} is invalid!".format(args.optimizer))
    return optimizer


