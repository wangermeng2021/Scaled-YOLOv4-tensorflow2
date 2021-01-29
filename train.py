
import tensorflow as tf
from model.yolov4 import Yolov4
from model.losses import yolov3_loss
from utils.optimizers import yolov3_optimizers
from utils.eager_coco_map import EagerCocoMap
from generator.generator_builder import get_generator
import time
import argparse
import sys
import os
from tqdm import tqdm
import logging
from utils.lr_scheduler import get_lr_scheduler
logging.getLogger().setLevel(logging.INFO)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
    #training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--start-eval-epoch', default=150, type=int)
    parser.add_argument('--eval-epoch-interval', default=1)
    #model
    parser.add_argument('--model-type', default='p5', help="choices=['p5','p6','p7']")
    parser.add_argument('--pretrained-weights', default='pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain',help="Path to a pretrain weights.")
    parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
    #loss
    parser.add_argument('--box-regression-loss', default='diou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
    parser.add_argument('--focal-alpha', default= 0.25)
    parser.add_argument('--focal-gamma', default=2.0)
    parser.add_argument('--ignore-thr', default=0.7)
    parser.add_argument('--reg-losss-weight', default=0.05)
    parser.add_argument('--obj-losss-weight', default=1.0)
    parser.add_argument('--cls-losss-weight', default=0.5)
    #dataset
    parser.add_argument('--num-classes', default=12)
    parser.add_argument('--class-names', default='chess.names')
    parser.add_argument('--dataset', default='dataset/chess_voc')
    parser.add_argument('--dataset-type', default='voc', help="voc,coco")
    parser.add_argument('--voc-train-set', default=[('dataset_1', 'train')],help="VOC dataset:[(VOC2007, 'trainval'), (VOC2012, 'trainval')]")
    parser.add_argument('--voc-valid-set', default=[('dataset_1', 'val')],help="VOC dataset:[(VOC2007, 'test')]")
    parser.add_argument('--voc-skip-difficult', default=True)

    '''
    coco dataset directory:
        annotations/instances_train2017.json
        annotations/instances_val2017.json
        images/train2017
        images/val2017
    '''
    parser.add_argument('--coco-train-set', default='train2017')
    parser.add_argument('--coco-valid-set', default='val2017')

    parser.add_argument('--augment', default='mosaic',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
    parser.add_argument('--multi-scale', default='352',help="Input data shapes for training, use 320+32*i(i>=0)")#896
    parser.add_argument('--max-box-num-per-image', default=100)
    #optimizer
    parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', default=5e-4)
    #lr scheduler
    parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3)
    parser.add_argument('--lr-decay', default=0.1)
    parser.add_argument('--lr-decay-epoch', default=[160, 180], type=int)
    parser.add_argument('--warmup-epochs', default=0)
    parser.add_argument('--warmup-lr', default=1e-4)
    #postprocess
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2)
    parser.add_argument('--score-threshold', default=0.5)
    #anchor
    parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2)
    parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0)

    parser.add_argument('--label-smooth', default=0.0)
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--accumulated-gradient-num', default=1)

    return parser.parse_args(args)

def main(args):
    train_generator, val_generator, pred_generator = get_generator(args)
    model = Yolov4(args, training=True)
    if args.pretrained_weights:
        if args.model_type == "p5":
            cur_num_classes = args.num_classes
            args.num_classes = 80
            pretrain_model = Yolov4(args, training=True)
            pretrain_model.load_weights(args.pretrained_weights).expect_partial()
            for layer in model.layers:
                if not layer.get_weights():
                    continue
                if 'yolov3_head' in layer.name:
                    continue
                layer.set_weights(pretrain_model.get_layer(layer.name).get_weights())
            args.num_classes = cur_num_classes
            logging.info("Load weight successfully!")
        else:
            logging.info("pretrain weight currently support only p5!")

    num_model_outputs = {"p5":3,"p6":4,"p7":5}
    loss_fun = [yolov3_loss(args, grid_index) for grid_index in range(num_model_outputs[args.model_type])]
    lr_scheduler = get_lr_scheduler(args)
    optimizer = yolov3_optimizers(args)

    start_time = time.perf_counter()
    coco_map = EagerCocoMap(pred_generator, model, args)
    max_coco_map = 0
    max_coco_map_epoch = 0
    accumulate_num = args.accumulated_gradient_num
    accumulate_index = 0
    accum_gradient = [tf.Variable(tf.zeros_like(this_var)) for this_var in model.trainable_variables]

    #training
    for epoch in range(args.epochs):
        lr = lr_scheduler(epoch)
        optimizer.learning_rate.assign(lr)
        remaining_epoches = args.epochs - epoch - 1
        epoch_start_time = time.perf_counter()
        train_loss = 0
        train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
        for batch_index, (batch_imgs, batch_labels)  in train_generator_tqdm:
            with tf.GradientTape() as tape:
                model_outputs = model(batch_imgs, training=True)
                data_loss = 0
                for output_index,output_val in enumerate(model_outputs):
                    loss = loss_fun[output_index](batch_labels[output_index], output_val)
                    data_loss += tf.reduce_sum(loss)

                total_loss = data_loss + args.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'batch_normalization' not in v.name])
            grads = tape.gradient(total_loss, model.trainable_variables)
            accum_gradient = [acum_grad.assign_add(grad) for acum_grad, grad in zip(accum_gradient, grads)]

            accumulate_index += 1
            if accumulate_index == accumulate_num:
                optimizer.apply_gradients(zip(accum_gradient, model.trainable_variables))
                accum_gradient = [ grad.assign_sub(grad) for grad in accum_gradient]
                accumulate_index = 0
            train_loss += total_loss
            train_generator_tqdm.set_description(
                "epoch:{}/{},train_loss:{:.4f},lr:{:.6f}".format(epoch, args.epochs,
                                                                                 train_loss/(batch_index+1),
                                                                                 optimizer.learning_rate.numpy()))
        train_generator.on_epoch_end()

        #evaluation
        if epoch >= args.start_eval_epoch:
            if epoch % args.eval_epoch_interval == 0:
                summary_metrics = coco_map.eval()
                if summary_metrics['Precision/mAP@.50IOU'] > max_coco_map:
                    max_coco_map = summary_metrics['Precision/mAP@.50IOU']
                    max_coco_map_epoch = epoch
                    model.save_weights(os.path.join(args.checkpoints_dir, 'scaled_yolov4_best_{}_{:.3f}'.format(max_coco_map_epoch, max_coco_map)))
            logging.info("max_coco_map:{},epoch:{}".format(max_coco_map,max_coco_map_epoch))

        cur_time = time.perf_counter()
        one_epoch_time = cur_time - epoch_start_time
        logging.info("time elapsed: {:.3f} hour, time left: {:.3f} hour".format((cur_time-start_time)/3600,remaining_epoches*one_epoch_time/3600))

    # model.save(os.path.join(args.checkpoints_dir, 'best_model_{}_{:.3f}'.format(max_coco_map_epoch, max_coco_map)))
    logging.info("Training is finished!")
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
