
import tensorflow as tf
from model.yolov4_tiny import Yolov4_tiny
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
from tensorboard import program
import numpy as np
import webbrowser
import logging
from utils.lr_scheduler import get_lr_scheduler
logging.getLogger().setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.setbufsize(1e7)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
    #save model
    parser.add_argument('--output-model-dir', default='./output_model')
    #training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--start-eval-epoch', default=10, type=int)
    parser.add_argument('--eval-epoch-interval', default=1)
    #model
    parser.add_argument('--model-type', default='tiny', help="choices=['tiny','p5','p6','p7']")
    parser.add_argument('--use-pretrain', default=False, type=bool)
    parser.add_argument('--tiny-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_tiny_coco_pretrain/coco_pretrain')
    parser.add_argument('--p5-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain')
    parser.add_argument('--p6-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p6_coco_pretrain/coco_pretrain')
    parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
    #loss
    parser.add_argument('--box-regression-loss', default='ciou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
    parser.add_argument('--focal-alpha', default= 0.25)
    parser.add_argument('--focal-gamma', default=2.0)
    parser.add_argument('--ignore-thr', default=0.7)
    parser.add_argument('--reg-losss-weight', default=0.05)
    parser.add_argument('--obj-losss-weight', default=1.0)
    parser.add_argument('--cls-losss-weight', default=0.5)
    #dataset
    parser.add_argument('--dataset-type', default='voc', help="voc,coco")
    parser.add_argument('--num-classes', default=20)
    parser.add_argument('--class-names', default='voc.names', help="voc.names,coco.names")
    parser.add_argument('--dataset', default='/home/wangem1/dataset/VOC2007_2012')#
    parser.add_argument('--voc-train-set', default='VOC2007,trainval,VOC2012,trainval')
    parser.add_argument('--voc-val-set', default='VOC2007,test')
    parser.add_argument('--voc-skip-difficult', default=True)
    parser.add_argument('--coco-train-set', default='train2017')
    parser.add_argument('--coco-valid-set', default='val2017')
    '''
    voc dataset directory:
        VOC2007
                Annotations
                ImageSets
                JPEGImages
        VOC2012
                Annotations
                ImageSets
                JPEGImages
    coco dataset directory:
        annotations/instances_train2017.json
        annotations/instances_val2017.json
        images/train2017
        images/val2017
    '''
    parser.add_argument('--augment', default='mosaic',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
    parser.add_argument('--multi-scale', default='416',help="Input data shapes for training, use 320+32*i(i>=0)")#896
    parser.add_argument('--max-box-num-per-image', default=100)
    #optimizer
    parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', default=5e-4)
    #lr scheduler
    parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[160, 180])
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-6, type=float)
    #postprocess
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
    parser.add_argument('--nms-score-threshold', default=0.01, type=float)
    #anchor
    parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
    parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0, type=float)

    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--accumulated-gradient-num', default=1, type=int)

    parser.add_argument('--tensorboard', default=True, type=bool)

    return parser.parse_args(args)

def main(args):
    train_generator, _, pred_generator = get_generator(args)

    if args.model_type == "tiny":
        model = Yolov4_tiny(args, training=True)
        if args.use_pretrain:
            if len(os.listdir(os.path.dirname(args.tiny_coco_pretrained_weights))) != 0:
                cur_num_classes = int(args.num_classes)
                args.num_classes = 80
                pretrain_model = Yolov4_tiny(args, training=True)
                pretrain_model.load_weights(args.tiny_coco_pretrained_weights).expect_partial()
                for layer in model.layers:
                    if not layer.get_weights():
                        continue
                    if 'yolov3_head' in layer.name:
                        continue
                    layer.set_weights(pretrain_model.get_layer(layer.name).get_weights())
                args.num_classes = cur_num_classes
                print("Load {} weight successfully!".format(args.model_type))
            else:
                raise ValueError("pretrained_weights directory is empty!")
    elif args.model_type == "p5":
        model = Yolov4(args, training=True)
        if args.use_pretrain:
            if len(os.listdir(os.path.dirname(args.p5_coco_pretrained_weights)))!=0:
                cur_num_classes = int(args.num_classes)
                args.num_classes = 80
                pretrain_model = Yolov4(args, training=True)
                pretrain_model.load_weights(args.p5_coco_pretrained_weights).expect_partial()
                for layer in model.layers:
                    if not layer.get_weights():
                        continue
                    if 'yolov3_head' in layer.name:
                        continue
                    layer.set_weights(pretrain_model.get_layer(layer.name).get_weights())
                args.num_classes = cur_num_classes
                print("Load {} weight successfully!".format(args.model_type))
            else:
                raise ValueError("pretrained_weights directory is empty!")
    elif args.model_type == "p6":
        model = Yolov4(args, training=True)
        if args.use_pretrain:
            if len(os.listdir(os.path.dirname(args.p6_coco_pretrained_weights))) != 0:
                cur_num_classes = int(args.num_classes)
                args.num_classes = 80
                pretrain_model = Yolov4(args, training=True)
                pretrain_model.load_weights(args.p6_coco_pretrained_weights).expect_partial()
                for layer in model.layers:
                    if not layer.get_weights():
                        continue
                    if 'yolov3_head' in layer.name:
                        continue
                    layer.set_weights(pretrain_model.get_layer(layer.name).get_weights())
                args.num_classes = cur_num_classes
                print("Load {} weight successfully!".format(args.model_type))
            else:
                raise ValueError("pretrained_weights directory is empty!")
    else:
        model = Yolov4(args, training=True)
        print("pretrain weight currently don't support p7!")
    num_model_outputs = {"tiny":2, "p5":3,"p6":4,"p7":5}
    loss_fun = [yolov3_loss(args, grid_index) for grid_index in range(num_model_outputs[args.model_type])]
    lr_scheduler = get_lr_scheduler(args)
    optimizer = yolov3_optimizers(args)

    print("loading dataset...")
    start_time = time.perf_counter()
    coco_map = EagerCocoMap(pred_generator, model, args)
    max_coco_map = -1
    max_coco_map_epoch = -1
    accumulate_num = args.accumulated_gradient_num
    accumulate_index = 0
    accum_gradient = [tf.Variable(tf.zeros_like(this_var)) for this_var in model.trainable_variables]

    best_weight_path = ""

    #tensorboard
    open_tensorboard_url = False
    os.system('rm -rf ./logs/')
    train_writer = tf.summary.create_file_writer("logs/train")
    mAP_writer = tf.summary.create_file_writer("logs/mAP")
    if args.tensorboard:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', 'logs','--reload_interval','15'])
        url = tb.launch()
        print("Tensorboard engine is running at {}".format(url))

    #training
    for epoch in range(int(args.epochs)):
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

        with train_writer.as_default():
            tf.summary.scalar("train_loss", train_loss/len(train_generator), step=epoch)
            train_writer.flush()

        #evaluation
        if epoch >= args.start_eval_epoch:
            if epoch % args.eval_epoch_interval == 0:
                summary_metrics = coco_map.eval()
                if summary_metrics['Precision/mAP@.50IOU'] > max_coco_map:
                    max_coco_map = summary_metrics['Precision/mAP@.50IOU']
                    max_coco_map_epoch = epoch
                    best_weight_path = os.path.join(args.checkpoints_dir, 'scaled_yolov4_best_{}_{:.3f}'.format(max_coco_map_epoch, max_coco_map))
                    model.save_weights(best_weight_path)

                print("max_coco_map:{},epoch:{}".format(max_coco_map,max_coco_map_epoch))
                with mAP_writer.as_default():
                    tf.summary.scalar("mAP@0.5", summary_metrics['Precision/mAP@.50IOU'], step=epoch)
                    mAP_writer.flush()
        cur_time = time.perf_counter()
        one_epoch_time = cur_time - epoch_start_time
        print("time elapsed: {:.3f} hour, time left: {:.3f} hour".format((cur_time-start_time)/3600,remaining_epoches*one_epoch_time/3600))

        if epoch>0 and not open_tensorboard_url:
            open_tensorboard_url = True
            webbrowser.open(url,new=1)

    print("Training is finished!")
    #save model
    print("Exporting model...")
    if args.output_model_dir and best_weight_path:
        tf.keras.backend.clear_session()
        if args.model_type == "tiny":
            model = Yolov4_tiny(args, training=False)
        else:
            model = Yolov4(args, training=False)
        model.load_weights(best_weight_path)
        model.save(os.path.join(args.output_model_dir,os.path.basename(args.class_names).split('.')[-2], 'best_model_{}_{:.3f}'.format(args.model_type, max_coco_map),'1'))

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
