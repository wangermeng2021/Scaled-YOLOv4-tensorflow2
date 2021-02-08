
import tensorflow as tf
from model.yolov4 import Yolov4

# from model.losses import yolov3_loss
# from model.optimizers import yolov3_optimizers
# from utils.eager_coco_map import EagerCocoMap
# from model.callbacks import warmup_lr_scheduler
# from data.generator import get_generator
import time
import argparse
import os




def get_sorted_layer_name(mode='p5'):

    if mode == 'p5':
        loop_nums = [1, 3, 15, 15, 7]
    elif mode == 'p6':
        loop_nums = [1, 3, 15, 15, 7, 7]
    elif mode == 'p7':
        loop_nums = [1, 3, 15, 15, 7, 7, 7]
    else:
        raise ValueError("{} is invalid!".format(mode))

    backbone_block_num = len(loop_nums)
    layer_names = []

    layer_names.append('first_block_conv2d')
    layer_names.append('first_block_batch_normalization')

    for block_index in range(backbone_block_num):

        layer_names.append('backbone_block_{}_0_conv2d'.format(block_index))

        layer_names.append('backbone_block_{}_0_batch_normalization'.format(block_index))

        layer_names.append('backbone_block_{}_1_right_branch_0_conv2d'.format(block_index))
        layer_names.append('backbone_block_{}_1_right_branch_0_batch_normalization'.format(block_index))

        layer_names.append('backbone_block_{}_1_left_branch_conv2d'.format(block_index))

        layer_names.append('backbone_block_{}_1_right_branch_1_conv2d'.format(block_index))

        layer_names.append('backbone_block_{}_1_foot_conv2d'.format(block_index))
        layer_names.append('backbone_block_{}_1_foot_batch_normalization'.format(block_index))

        layer_names.append('backbone_block_{}_1_concat_batch_normalization'.format(block_index))

        for res_index in range(loop_nums[block_index]):
            layer_names.append('backbone_block_{}_1_right_branch_res_{}_conv2d'.format(block_index,res_index*2))
            layer_names.append('backbone_block_{}_1_right_branch_res_{}_batch_normalization'.format(block_index,res_index*2))
            layer_names.append('backbone_block_{}_1_right_branch_res_{}_conv2d'.format(block_index,res_index*2+1))
            layer_names.append('backbone_block_{}_1_right_branch_res_{}_batch_normalization'.format(block_index,res_index*2+1))
    #spp
    layer_names.append('spp_right_branch_0_conv2d')
    layer_names.append('spp_right_branch_0_batch_normalization')
    layer_names.append('spp_left_branch_conv2d')
    layer_names.append('spp_right_branch_1_conv2d')
    layer_names.append('spp_right_branch_1_batch_normalization')
    layer_names.append('spp_right_branch_2_conv2d')
    layer_names.append('spp_right_branch_2_batch_normalization')
    layer_names.append('spp_right_branch_3_conv2d')
    layer_names.append('spp_right_branch_3_batch_normalization')
    layer_names.append('spp_right_branch_4_conv2d')
    layer_names.append('spp_right_branch_4_batch_normalization')
    layer_names.append('spp_concat_batch_normalization')
    layer_names.append('spp_foot_conv2d')
    layer_names.append('spp_foot_batch_normalization')

    head_down_block_num = backbone_block_num - 3
    head_up_block_num = backbone_block_num - 3
    head_down_loop_nums = head_up_loop_nums = 3
    for block_index in range(head_down_block_num):
        layer_names.append('head_down_block_{}_0_conv2d'.format(block_index))
        layer_names.append('head_down_block_{}_0_batch_normalization'.format(block_index))
        layer_names.append('head_down_block_{}_1_conv2d'.format(block_index))
        layer_names.append('head_down_block_{}_1_batch_normalization'.format(block_index))
        layer_names.append('head_down_block_{}_2_head_conv2d'.format(block_index))
        layer_names.append('head_down_block_{}_2_head_batch_normalization'.format(block_index))
        layer_names.append('head_down_block_{}_2_left_branch_conv2d'.format(block_index))
        layer_names.append('head_down_block_{}_2_foot_conv2d'.format(block_index))
        layer_names.append('head_down_block_{}_2_foot_batch_normalization'.format(block_index))
        layer_names.append('head_down_block_{}_2_concat_batch_normalization'.format(block_index))
        for res_index in range(head_down_loop_nums):
            layer_names.append('head_down_block_{}_2_right_branch_res_{}_conv2d'.format(block_index, res_index*2))
            layer_names.append('head_down_block_{}_2_right_branch_res_{}_batch_normalization'.format(block_index, res_index*2))
            layer_names.append('head_down_block_{}_2_right_branch_res_{}_conv2d'.format(block_index, res_index*2+1))
            layer_names.append('head_down_block_{}_2_right_branch_res_{}_batch_normalization'.format(block_index, res_index*2+1))
    layer_names.append('yolov3_head_0_0_conv2d')
    layer_names.append('yolov3_head_0_0_batch_normalization')

    for block_index in range(head_up_block_num):
        layer_names.append('head_up_block_{}_0_conv2d'.format(block_index))
        layer_names.append('head_up_block_{}_0_batch_normalization'.format(block_index))
        layer_names.append('head_up_block_{}_1_head_conv2d'.format(block_index))
        layer_names.append('head_up_block_{}_1_head_batch_normalization'.format(block_index))
        layer_names.append('head_up_block_{}_1_left_branch_conv2d'.format(block_index))
        layer_names.append('head_up_block_{}_1_foot_conv2d'.format(block_index))
        layer_names.append('head_up_block_{}_1_foot_batch_normalization'.format(block_index))
        layer_names.append('head_up_block_{}_1_concat_batch_normalization'.format(block_index))
        for res_index in range(head_up_loop_nums):
            layer_names.append('head_up_block_{}_1_right_branch_res_{}_conv2d'.format(block_index, res_index*2))
            layer_names.append('head_up_block_{}_1_right_branch_res_{}_batch_normalization'.format(block_index, res_index*2))
            layer_names.append('head_up_block_{}_1_right_branch_res_{}_conv2d'.format(block_index, res_index*2+1))
            layer_names.append('head_up_block_{}_1_right_branch_res_{}_batch_normalization'.format(block_index, res_index*2+1))

        layer_names.append('yolov3_head_{}_0_conv2d'.format(block_index+1))
        layer_names.append('yolov3_head_{}_0_batch_normalization'.format(block_index+1))



    detect_block_num = backbone_block_num-2
    for index in range(detect_block_num):
        layer_names.append('yolov3_head_{}_1_conv2d'.format(index))

    # for index in range(detect_block_num):
    #     layer_names.append('yolov3_head_{}_0_conv2d'.format(index))
    #     layer_names.append('yolov3_head_{}_0_batch_normalization'.format(index))
    #     layer_names.append('yolov3_head_{}_1_conv2d'.format(index))
    return layer_names

layer_names = get_sorted_layer_name()
for i in layer_names:
    print(i)


exit()









os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using snapmix .')
    parser.add_argument('--model_type', default='p5', help="choices=['p5','p6','p7']")

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=16, type=int)

    parser.add_argument('--dataset-root-dir', default='/home/wangem1/dataset/VOC2007&2012', type=str, help="voc,coco")
    parser.add_argument('--dataset_type', default='voc', type=str, help="voc,coco")
    parser.add_argument('--voc_train_set', default=[(2007, 'trainval'), (2012, 'trainval')])
    parser.add_argument('--voc_valid_set', default=[(2007, 'test')])
    parser.add_argument('--voc_skip_difficult', default=True)
    parser.add_argument('--coco_train_set', default='train')
    parser.add_argument('--coco_valid_set', default='valid')
    parser.add_argument('--num-classes', default=80, help="choices=['p5','p6','p7']")
    parser.add_argument('--class_names', default='voc.names', help="choices=['p5','p6','p7']")

    parser.add_argument('--augment', default='rand_augment', type=str, help="choices=[random_crop,'mosaic','only_flip_left_right',None]")

    parser.add_argument('--multi-scale', default=[416], help="choices=['p5','p6','p7']")

    parser.add_argument('--start-val-epoch', default=50, type=int)

    parser.add_argument('--optimizer', default='adam', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9, help="choices=[sgd,'p6','p7']")
    parser.add_argument('--nesterov', default=True, help="choices=[sgd,'p6','p7']")
    parser.add_argument('--weight_decay', default=True, help="")

    parser.add_argument('--lr-scheduler', default='step', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[80, 150, 180], type=int)
    parser.add_argument('--warmup-lr', default=1e-4, type=float)
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)

    parser.add_argument('--transfer-type', default='csp_darknet53_and_pan', help="choices=['p5','p6','p7']")#choices=['csp_darknet53','csp_darknet53_and_pan',None]
    parser.add_argument('--pretrained-weights', default='p5', help="choices=['p5','p6','p7']")
    parser.add_argument('--output-checkpoints-dir', default='p5', help="choices=['p5','p6','p7']")







    parser.add_argument('--box_regression_loss', default='diou')#{'mse','bce_and_l1loss','diou','ciou'}
    parser.add_argument('--classification_loss', default='bce', help="choices=['p5','p6','p7']")#,#choices=['ce','bce','focal']
    parser.add_argument('--object_loss', default='bce', help="choices=['p5','p6','p7']")
    parser.add_argument('--focal_alpha', default= 0.25, help="choices=['p5','p6','p7']")
    parser.add_argument('--focal_gamma', default=2.0, help="choices=['p5','p6','p7']")
    parser.add_argument('--ignore_thr', default=0.7, help="choices=['p5','p6','p7']")

    #postprocess
    parser.add_argument('--nms', default=[416], help="choices=['p5','p6','p7']")
    parser.add_argument('--max-boxes-num', default=1000, help="choices=['p5','p6','p7']")
    parser.add_argument('--nms-iou-threshold', default=1000, help="choices=['p5','p6','p7']")
    parser.add_argument('--score-threshold', default=[416], help="choices=['p5','p6','p7']")
    parser.add_argument('--pre-nms-num-boxes', default=1000, help="choices=['p5','p6','p7']")

    parser.add_argument('--label-smooth', default=0.1, help="choices=['p5','p6','p7']")
    parser.add_argument('--scales-x-y', default=[2., 2., 2.], help="choices=['p5','p6','p7']")
    parser.add_argument('--multi-anchor-iou_thr', default=0.1, help="choices=['p5','p6','p7']")

    return parser.parse_args(args)

import numpy as np
my_weight = np.load('my_weight.npy',allow_pickle=True)
# print(my_weight['first_block_conv2d'])
# print(my_weight.item())

def main(args):

    # train_dataset, valid_dataset, pred_dataset = get_generator(args)

    model = Yolov4(args, training=False)

    for name in my_weight.item():
        print(name)
        model.get_layer(name).set_weights(my_weight.item()[name])

    exit()
    aaa = []
    # model.summary()
    # for l1 in model.layers:
    #     if isinstance(l1, tf.keras.Model):
    #         for l2 in l1.layers:
    #             if l2.name not in layer_names:
    #                 print(l2.name)
    for l1 in layer_names:
        print(l1, model.get_layer(l1).get_weights()[0].shape)
        aaa.append(model.get_layer(l1).get_weights()[0].shape)
    # layer_names


    aaa = np.array(aaa, dtype=object)
    # print(aaa)
    np.save('2.npy', aaa)
    exit()



    # freeze_all(model)
    conv_num = 0
    bn_num = 0
    for l1 in model.layers:
        if isinstance(l1, tf.keras.Model):
            # print(l1.name,"-----------------------")


            for l2 in l1.layers:
                print(l2.name)
                if 'conv2d' in l2.name:
                    # s1 = l2.get_weights()
                    # print("/////////////////////////////////////////")
                    # print(l1.name,l2.name,l2.output.shape)
                    # print(s1[0].shape)
                    # if 'backbone_block_4' in l2.name:
                    #     conv_num+=1
                    conv_num += 1
                    # print(l2.name, "-----------------------")

                elif 'batch_normalization' in l2.name:

                    # print(l1.name,l2.name,l2.output.shape)
                    s1 = l2.get_weights()
                    print(s1)
                    # pass
                    # print(filters, biases )
                    # if 'backbone_block_4' in l2.name:
                    #     bn_num+=1
                    bn_num += 1
                    # print(l2.name, "-----------------------")
                # else:
                #     print(l2.name)
                # print("conv_num:", conv_num)
                # print("bn_num:", bn_num)
        else:
            # print(l1.name,"bbbbbbbbbbbbbbbbbbbb")
            pass
    print("conv_num:",conv_num)
    print("bn_num:", bn_num)
    exit()
    if CFG['train']['transfer_type']!=None:
        coco_pretrained_model = Yolov4(CFG, training=True, load_coco_pretrained_weight=True)
        coco_pretrained_model.load_weights(CFG['train']['pretrained_weights_path']).expect_partial()
        if CFG['train']['transfer_type'] == 'csp_darknet53':
            model.get_layer('csp_darknet53').set_weights(coco_pretrained_model.get_layer('csp_darknet53').get_weights())

        elif CFG['train']['transfer_type'] == 'csp_darknet53_and_pan':
            model.get_layer('csp_darknet53').set_weights(coco_pretrained_model.get_layer('csp_darknet53').get_weights())
            model.get_layer('pan').set_weights(coco_pretrained_model.get_layer('pan').get_weights())

    if CFG['train']['mode'] == 'keras_fit':
        model.compile(optimizer=yolov3_optimizers(CFG),
                      loss=[yolov3_loss(CFG, grid_index) for grid_index in range(3)])
        callbacks = get_all_callbacks(CFG)
        model.fit(train_dataset, validation_data=valid_dataset,
                  batch_size=CFG['train']['batch_size'],
                  epochs=CFG['train']['epochs'],
                  callbacks=callbacks)
        # model.save_weights("/home/wangem1/yolov4/yolov4/checkpoints/1.tf")
        model.save("checkpoints/best_model/1")
    elif CFG['train']['mode'] == 'tf_eager':

            max_mAP = 0

            start_time = time.perf_counter()
            coco_map = EagerCocoMap(pred_dataset, model)

            optimizer = yolov3_optimizers(CFG)

            accumulate_num = CFG['train']['accumulated_gradient_num']
            sum_num = 0

            accum_gradient = [tf.Variable(tf.zeros_like(this_var)) for this_var in model.trainable_variables]

            for epoch_index in range(CFG['train']['epochs']):
                remaining_epoches = CFG['train']['epochs'] - epoch_index - 1
                print('Epoch %d is running...' % (epoch_index))
                epoch_start_time = time.perf_counter()
                for train_input in train_dataset:
                    batch_start_time = time.perf_counter()
                    with tf.GradientTape() as tape:
                        model_outputs = model(train_input[0], training=True)

                        data_y_true = train_input[1]
                        loss = [yolov3_loss(CFG, grid_index) for grid_index in range(3)]
                        loss0 = loss[0](data_y_true[0], model_outputs[0])
                        loss1 = loss[1](data_y_true[1], model_outputs[1])
                        loss2 = loss[2](data_y_true[2], model_outputs[2])
                        total_loss = tf.reduce_sum(loss0) + tf.reduce_sum(loss1) + tf.reduce_sum(loss2)

                    grads = tape.gradient(total_loss, model.trainable_variables)
                    accum_gradient = [acum_grad.assign_add(grad) for acum_grad, grad in zip(accum_gradient, grads)]

                    sum_num += 1
                    if sum_num == accumulate_num:
                        # accum_gradient = [this_grad / accumulate_num for this_grad in accum_gradient]
                        optimizer.apply_gradients(zip(accum_gradient, model.trainable_variables))

                        sum_num = 0
                        accum_gradient = [ grad.assign_sub(grad) for grad in accum_gradient]

                    batch_end_time = time.perf_counter()
                    # print("time elapsed: %.3f(hour), time left: %.3f(hour)" % (
                    # (batch_end_time - start_time) / 3600., (batch_end_time - batch_start_time)*train_dataset.__len__()*remaining_epoches / 3600.))
                    #
                    # print('train_loss:%.2f' % (total_loss))

                lr = warmup_lr_scheduler(CFG)(epoch_index, 0)
                optimizer.learning_rate.assign(lr)

                print('Epoch %d train_loss:%.3f - lr:%f' % (epoch_index, total_loss, lr.numpy()))

                #time elapsed
                cur_time = time.perf_counter()
                one_epoch_time = cur_time - epoch_start_time

                print("time elapsed: %.3f(hour), time left: %.3f(hour)" %((cur_time-start_time)/3600,remaining_epoches*one_epoch_time/3600))
                # print("time elapsed: %d(second), time left: %d(second), estimated finish time: %d(hour)" %(time.perf_counter()- epoch_start_time))

                if (epoch_index+1) % CFG['eval']['eval_epoch_interval'] != 0:
                    pass
                    # loss = [yolov3_loss(CFG, grid_index) for grid_index in range(3)]
                    # for valid_input in valid_dataset:
                    #     model_outputs = model(valid_input[0], training=False)
                    #     data_y_true = valid_input[1]
                    #     loss0 = loss[0](data_y_true[0], model_outputs[0])
                    #     loss1 = loss[1](data_y_true[1], model_outputs[1])
                    #     loss2 = loss[2](data_y_true[2], model_outputs[2])
                    #     total_loss = tf.reduce_sum(loss0) + tf.reduce_sum(loss1) + tf.reduce_sum(loss2)
                    #     # print('Epoch %d valid_loss:%.2f'%(epoch_index, total_loss))
                    # print('Epoch %d valid_loss:%.2f' % (epoch_index, total_loss))
                else:
                    summary_metrics = coco_map.eval()
                    if summary_metrics['Precision/mAP@.50IOU'] > max_mAP:
                        max_mAP = summary_metrics['Precision/mAP@.50IOU']
                        model.save_weights('checkpoints/best_model_%d_%.3f' % (epoch_index, max_mAP))

import sys
if __name__== "__main__":
    args = parse_args(sys.argv[1:])
    main(args)