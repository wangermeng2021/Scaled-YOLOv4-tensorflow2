
# Scaled-YOLOv4-tensorflow2
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)

A Tensorflow2.x implementation of Scaled-YOLOv4 as described in [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)


## Update Log
[2021-07-02]:
* Add support for: Exponential moving average decay for variables. Improve mAP from 0.985 to 0.990 on Chess Pieces dataset.

[2021-06-29]:

Major Features and Improvements:
* Add support for: Sharpness-Aware Minimization(SAM_sgd,SAM_adam).

Bug Fixes and Changes:
* Fix the nan loss error when using adam optimizer
* Set default optimizer as SAM_adam
* Change default running mode from 'fit' to 'eager mode'

[2021-06-27] Add support for: resuming training from checkpoints.<br><br>
[2021-02-21] Add support for: model.fit(dramatic improvement in GPU utilization); online coco evaluation callback; change default optimizer from sgd to adam <br><br>
[2021-02-11] Add support for: one-click deployment using tensorflow Serving(very fast)<br><br>
[2021-01-29] Add support for: mosaic,ssd_random_crop<br><br>
[2021-01-25] Add support for: ciou loss,hard-nms,DIoU-nms,label_smooth,transfer learning,tensorboard<br><br>
[2021-01-23] Add support for: scales_x_y/eliminate grid sensitivity,accumulate gradients for using big batch size,focal loss,diou loss<br><br>
[2021-01-16] Add support for: warmup,Cosine annealing scheduler,Eager mode training with tf.GradientTape,support voc/coco dataset format<br><br>
[2021-01-10] Add support for: yolov4-tiny,yolov4-large p5/p6/p7,online coco evaluation,multi scale training<br><br>

## Demo
ScaledYOLOv4_p5_detection_result:

![pothole_p5_detection_3.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/pothole_p5_detection_3.png)
![chess_p5_detection.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/chess_p5_detection.png)

ScaledYOLOv4_tiny_detection_result:

![safehat_tiny_detection_1.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/safehat_tiny_detection_1.png)
![safehat_tiny_detection_2.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/safehat_tiny_detection_2.png)

## Installation
###  1. Clone project
  ``` 
  git clone https://github.com/wangermeng2021/Scaled-YOLOv4-tensorflow2.git
  cd Scaled-YOLOv4-tensorflow2
  ```
###   2. Install environment
* install tesnorflow ( skip this step if it's already installed,test environment:tensorflow 2.4.0)
*     pip install -r requirements.txt


## Note:
I strongly recommend using voc dataset type(default dataset type), because my GPU is old, so coco dataset type is not fully tested.

## Training:
* Download Pre-trained p5 coco pretrain models and place it under directory 'pretrained/ScaledYOLOV4_p5_coco_pretrain' :<br>
   [https://drive.google.com/file/d/1glOCE3Y5Q5enW3rpVq3SmKDXzaKIw4YL/view?usp=sharing](https://drive.google.com/file/d/1glOCE3Y5Q5enW3rpVq3SmKDXzaKIw4YL/view?usp=sharing) <br>

* Download Pre-trained p6 coco pretrain models and place it under directory 'pretrained/ScaledYOLOV4_p6_coco_pretrain' :<br>
   [https://drive.google.com/file/d/1EymbpgiO6VkCCFdB0zSTv0B9yB6T9Fw1/view?usp=sharing](https://drive.google.com/file/d/1EymbpgiO6VkCCFdB0zSTv0B9yB6T9Fw1/view?usp=sharing) <br>

* Download Pre-trained tiny coco pretrain models and place it under directory 'pretrained/ScaledYOLOV4_tiny_coco_pretrain' :<br>
   [https://drive.google.com/file/d/1x15FN7jCAFwsntaMwmSkkgIzvHXUa7xT/view?usp=sharing](https://drive.google.com/file/d/1x15FN7jCAFwsntaMwmSkkgIzvHXUa7xT/view?usp=sharing) <br>




* For training on [Pothole dataset](https://public.roboflow.com/object-detection/pothole)(No need to download dataset,it's already included in project): <br>
  p5(single scale):
  ```
  python train.py --use-pretrain True --model-type p5 --dataset-type voc --dataset dataset/pothole_voc --num-classes 1 --class-names pothole.names  --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 200 --batch-size 4 --multi-scale 416 --augment ssd_random_crop 
  ```
  p5(multi scale):
  ```
  python train.py --use-pretrain True --model-type p5 --dataset-type voc --dataset dataset/pothole_voc --num-classes 1 --class-names pothole.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 200 --batch-size 4 --multi-scale 320,352,384,416,448,480,512 --augment ssd_random_crop 
  ```
* For training on [Chess Pieces dataset](https://public.roboflow.com/object-detection/chess-full)(No need to download dataset,it's already included in project): <br>
  tiny(single scale):
  ```
  python train.py --use-pretrain True --model-type tiny --dataset-type voc --dataset dataset/chess_voc --num-classes 12 --class-names chess.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 400 --batch-size 32 --multi-scale 416 --augment ssd_random_crop 
  ```
  tiny(multi scale):
  ```
  python train.py --use-pretrain True --model-type tiny --dataset-type voc --dataset dataset/chess_voc --num-classes 12 --class-names chess.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 400 --batch-size 32 --multi-scale 320,352,384,416,448,480,512 --augment ssd_random_crop

* For training with SAM_sgd on [Chess Pieces dataset](https://public.roboflow.com/object-detection/chess-full): <br>
  ```
  python train.py --optimizer SAM_sgd --use-pretrain True --model-type tiny --dataset-type voc --dataset dataset/chess_voc --num-classes 12 --class-names chess.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 400 --batch-size 32 --multi-scale 416 --augment ssd_random_crop 
  ```
* For training with ema(Exponential Moving Average) on [Chess Pieces dataset](https://public.roboflow.com/object-detection/chess-full): <br>
  ```
  python train.py --ema True --use-pretrain True --model-type tiny --dataset-type voc --dataset dataset/chess_voc --num-classes 12 --class-names chess.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 400 --batch-size 32 --multi-scale 416 --augment ssd_random_crop 
  ``` 
## Tensorboard visualization:
  * Navigate to [http://0.0.0.0:6006](http://0.0.0.0:6006)

## Evaluation results(GTX2080,mAP@0.5):

| model                                               | Chess Pieces | pothole |  VOC  | COCO |
|-----------------------------------------------------|--------------|---------|-------|------|
| Scaled-YoloV4-tiny(416)                             |     0.985    |         |       |      |
| Scaled-YoloV4-tiny(416)+ema                         |     0.990    |         |       |      |
| AlexeyAB's YoloV4(416)                              |              |  0.814  |       |      |
| Scaled-YoloV4-p5(416)                               |              |  0.826  |       |      |

* Evaluation on Pothole dataset: 
![tensorboard_pothole_p5.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/tensorboard_pothole_p5.png)
* Evaluation on chess dataset: 
![tensorboard_chess_tiny.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/tensorboard_chess_tiny.png)
## Detection
* For detection on Chess Pieces dataset:
  ```
  python3 detect.py --pic-dir images/chess_pictures --model-path output_model/best_model_tiny_0.985/1 --class-names dataset/chess.names --nms-score-threshold 0.1
  ```
  detection result:

  ![chess_p5_detection.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/chess_p5_detection.png)

* For detection on Pothole dataset:
  ```
  python3 detect.py --pic-dir images/pothole_pictures --model-path output_model/best_model_p5_0.827/1 --class-names dataset/pothole.names --nms-score-threshold 0.1
  ```
  detection result:

  ![pothole_p5_detection_2.png](https://github.com/wangermeng2021/ScaledYOLOv4-tensorflow2/blob/main/images/pothole_p5_detection_2.png)


## Customzied training
* Convert your dataset to Pascal VOC format(you can use labelImg to generate VOC format dataset)
* Generate class names file(such as xxx.names)
* 
  ```
  python train.py --use-pretrain True --model-type p5 --dataset-type voc --dataset your_dataset_root_dir --num-classes num_of_classes --class-names path_of_xxx.names --voc-train-set dataset_1,train --voc-val-set dataset_1,val  --epochs 200 --batch-size 8 --multi-scale 416  --augment ssd_random_crop 
  ```
## Deployment
TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments.it include two parts:clients and server, we can run them on one machine.<br>
* **Navigate to deployment directory:**
```
  cd  deployment/tfserving
```
* **Generate a docker image which contains your trained model (it will take minutes，you only have to run it one time):**
```
  ./gen_image --model-dir ScaledYOLOv4-tensorflow2/output_model/pothole/best_model_p5_0.811
```
* **Deploy model:**<br>
    * **Server side**( docker and nvidia-docker installed ):
	
        ` ./run_image `
	
    * **Client side**(no need to install tensorflow):<br>
        1. install client package

            `   pip install tfservingclient-1.0.0-cp37-cp37m-manylinux1_x86_64.whl   `


        2. predict images

            `   python demo.py --pic-dir xxxx --class-names xxx.names   `


## References
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/dmlc/gluon-cv](https://github.com/dmlc/gluon-cv)



