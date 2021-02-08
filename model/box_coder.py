
import tensorflow as tf
from utils.anchors import yolo_anchors

def box_decode(encoded_pred_boxes, args, grid_index):
    '''

    :param boxes: model's output box,shape is [batch,h,w,3,4]
    :param params:
    :param grid_index:choices=[0,1,2]
    :return: decoded boxes, shape is [batch,N,4]
    '''

    if args.model_type == 'tiny':
        stride = 16 * 2 ** grid_index
    else:
        stride = 8 * 2 ** grid_index

    (batch_size, grid_height, grid_width) = tf.shape(encoded_pred_boxes)[0:3]
    normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index],tf.dtypes.float32)/tf.cast(tf.shape(encoded_pred_boxes)[1:3]*stride,tf.dtypes.float32)

    grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
    grid_xy = tf.cast(tf.expand_dims(grid_xy, axis=-2),tf.dtypes.float32)

    scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
    decoded_pred_cxcy = (grid_xy + encoded_pred_boxes[..., 0:2]*scales_x_y - 0.5 * (scales_x_y - 1))/(grid_width,grid_height)

    decoded_pred_wh = (encoded_pred_boxes[..., 2:4]*2)**2 *normalized_anchors

    half_decoded_pred_wh = decoded_pred_wh/2
    decoded_x1y1x2y2 = tf.clip_by_value(tf.concat([decoded_pred_cxcy - half_decoded_pred_wh, decoded_pred_cxcy+half_decoded_pred_wh],axis=-1),0,1.)

    return tf.reshape(decoded_x1y1x2y2, [batch_size, -1, 4])
