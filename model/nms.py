
"""

"""
import numpy as np
import tensorflow as tf
class NonMaxSuppression():

    @staticmethod
    def hard_nms_tf(boxes, scores, params):
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=scores,
            max_output_size_per_class=params.nms_max_box_num,
            max_total_size=params.nms_max_box_num,
            iou_threshold=params.nms_iou_threshold,
            score_threshold=params.score_threshold
        )
        return boxes, scores, classes, valid_detections
    @staticmethod
    def soft_nms(boxes, iou_thr, score_thr=0.0):
        pass
    @staticmethod
    def hard_nms_np(batch_boxes, batch_scores, params):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """

        iou_threshold = params.nms_iou_threshold
        score_threshold =  params.score_threshold
        nms_max_box_num =  params.nms_max_box_num

        batch_classes = np.argmax(batch_scores, axis=-1)
        batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):
            # print(batch_result_boxes[0])
            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]

            classes = batch_classes[batch_index]

            valid_mask = scores > score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                # center_dist = np.sum(
                #     np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                #     axis=-1)
                # bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                # diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                # diou = iou_score - center_dist / diagonal_dist
                diou = iou_score
                valid_mask = diou <= iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            # result_boxes = np.array(result_boxes)
            # result_scores = np.array(result_scores)
            # result_classes = np.array(result_classes)
            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))


            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid

        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
    @staticmethod
    def diou_nms_np(batch_boxes, batch_scores, params):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """

        iou_threshold = params.nms_iou_threshold
        score_threshold = params.score_threshold
        nms_max_box_num = params.nms_max_box_num

        batch_classes = np.argmax(batch_scores, axis=-1)
        batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):
            # print(batch_result_boxes[0])
            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]

            classes = batch_classes[batch_index]

            valid_mask = scores > score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                center_dist = np.sum(
                    np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                    axis=-1)
                bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                diou = iou_score - center_dist / diagonal_dist
                # print(diou)
                valid_mask = diou <= iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))

            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid

        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
    @staticmethod
    def diou_nms_np_tta(batch_boxes, batch_scores, batch_classes,params):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """

        iou_threshold = params.nms_iou_threshold
        score_threshold = params.score_threshold
        nms_max_box_num = params.nms_max_box_num

        # batch_classes = np.argmax(batch_scores, axis=-1)
        # batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):
            # print(batch_result_boxes[0])
            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]

            classes = batch_classes[batch_index]

            valid_mask = scores > score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                center_dist = np.sum(
                    np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                    axis=-1)
                bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                diou = iou_score - center_dist / diagonal_dist
                # print(diou)
                valid_mask = diou <= iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))

            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid




        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
def yolov4_nms(args):
    if args.nms == 'hard_nms':
        return NonMaxSuppression.hard_nms_np
    elif args.nms == 'diou_nms':
        return NonMaxSuppression.diou_nms_np

    pass




# # # #test

#
# # a1 = np.random.choice(range(1,1000),[5, 6],replace=False)
# # r1 = np.random.randint(1,1000,[5,2])
# # a1[:, 2:4] = a1[:, 0:2]+r1
# # a1 = a1.astype(np.float32)
# # # a1 = np.expand_dims(a1, axis=0)
# #
# # boxes=np.array([[0.,0],[0.3,0.4]])
# # scores = np.array([0.8,0.1])
# # result = tf.image.non_max_suppression_overlaps(
# #     boxes, scores, 100, overlap_threshold=0.5,
# #     score_threshold=float('-inf'), name=None
# # )
# # print(result)
# # exit()
#
#
#
#
# # from config.param_config import CFG
# # a1 = np.random.choice(range(1,1000),[5, 6],replace=False)
# # r1 = np.random.randint(1,1000,[5,2])
# # a1[:, 2:4] = a1[:, 0:2]+r1
# # a1 = a1.astype(np.float32)
# # a1 = np.expand_dims(a1, axis=0)
# # s2 = NonMaxSuppression.diou_nms(a1[...,0:4], a1[...,4:6],CFG)
# # print(s2)
#
#
# # s1 = tf.keras.Input((5,))
# # s2 = NonMaxSuppression.diou_nms1(s1, -0.2)
# # model = tf.keras.Model(inputs = s1,outputs=s2)
# # model.predict([[1,1,100,100],[1,1,200,200]])
# # #
# # # tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# #
# from config.param_config import CFG
# i=0
# while i <2:
#     # np.random.seed(111223)
#
#     a1 = np.random.choice(range(1,1000),[1,6],replace=False)
#     r1 = np.random.randint(1,1000,[1,2])
#     a1[:, 2:4] = a1[:, 0:2]+r1
#     a1 = a1.astype(np.float32)
#
#     a11 = a1.copy()
#     a12 = np.expand_dims(a1.copy(),axis=0)
#
#     s1 = NonMaxSuppression.diou_nms_np(a11)
#     s2 = NonMaxSuppression.diou_nms_1(a12[...,0:4], a12[...,4:6], CFG)
#     # with tf.compat.v1.Session() as sess:
#     #     # s11 = sess.run(s1)
#     #     s22 = sess.run([s2])
#     try:
#         assert (s1[...,0:4]==s2.numpy()[0]).all()
#         # print(s1)
#         # print(s2)
#     except:
#         # print(a11)
#         # print(a11[np.argsort(a1[:,4])[::-1]])
#         print("nnnnnnnnnnnnnnnnnnnnnnnnn")
#         print(a12)
#         print(s1)
#         print(s2)
#     # print(s1)
#     # print(s2.numpy())
#     # print("sssssssssssssssss")
#     # # print(s1)
#     # # print(s2)
#     # if s1.shape[0]!=s2.shape[0]:
#     #     print("ssssssssssssssssssssssss")
#     i += 1
