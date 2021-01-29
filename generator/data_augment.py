
import os
import logging
import time
import warnings
import numpy as np
import cv2
from PIL import Image




def random_hsv(img,ratio=0.5):
    random_hsv_ratio = np.random.uniform(ratio, 1.0+ratio, 3)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_img = hsv_img.astype(np.uint16)*random_hsv_ratio
    hsv_img[..., 0] = np.clip(hsv_img[..., 0],0,180)
    hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)
    hsv_img[..., 2] = np.clip(hsv_img[..., 2], 0, 255)
    hsv_img = hsv_img.astype(np.uint8)
    cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR, dst=img)
def random_left_right_flip(img, boxes):
    # print(boxes)
    if np.random.random()<0.5:
        img[...] = np.fliplr(img)
        boxes[..., [2,0]] = img.shape[1]  - boxes[..., [0, 2]]
    # print(boxes)
    # return img,boxes
    # return img.copy(), boxes.copy()
# def random_affine(img, targets=(), degrees=0, translate=0, scale=0, shear=0, border=0):
def random_affine(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = np.random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = np.random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = np.random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = np.random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # x shear (deg)
    S[1, 0] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 0:4] = xy[i]

    return img, targets
def load_mosaic(batch_img, batch_boxes):
    # loads images in a mosaic
    h, w = batch_img[0].shape[0:2]
    four_boxes = []
    s = h
    xc, yc = [int(np.random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y


    four_img = np.full((s * 2, s * 2, batch_img[0].shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    four_img[y1a:y2a, x1a:x2a] = batch_img[0][y1b:y2b, x1b:x2b]
    padw = x1a - x1b
    padh = y1a - y1b

    new_batch_boxes = batch_boxes[0].copy()
    if new_batch_boxes.size > 0:
        new_batch_boxes[:, 0:4] = new_batch_boxes[:, 0:4]+np.tile([padw,padh],[2])
    four_boxes.append(new_batch_boxes)

    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    four_img[y1a:y2a, x1a:x2a] = batch_img[1][y1b:y2b, x1b:x2b]
    padw = x1a - x1b
    padh = y1a - y1b
    new_batch_boxes = batch_boxes[1].copy()
    if new_batch_boxes.size > 0:
        new_batch_boxes[:, 0:4] = new_batch_boxes[:, 0:4]+np.tile([padw,padh],[2])
    four_boxes.append(new_batch_boxes)

    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
    four_img[y1a:y2a, x1a:x2a] = batch_img[2][y1b:y2b, x1b:x2b]
    padw = x1a - x1b
    padh = y1a - y1b
    new_batch_boxes = batch_boxes[2].copy()
    if new_batch_boxes.size > 0:
        new_batch_boxes[:, 0:4] = new_batch_boxes[:, 0:4]+np.tile([padw,padh],[2])
    four_boxes.append(new_batch_boxes)

    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    four_img[y1a:y2a, x1a:x2a] = batch_img[3][y1b:y2b, x1b:x2b]
    padw = x1a - x1b
    padh = y1a - y1b
    new_batch_boxes = batch_boxes[3].copy()
    if new_batch_boxes.size > 0:
        new_batch_boxes[:, 0:4] = new_batch_boxes[:, 0:4]+np.tile([padw,padh],[2])
    four_boxes.append(new_batch_boxes)



    # Concat/clip labels
    if len(four_boxes):
        four_boxes = np.concatenate(four_boxes, 0)
        # np.clip(four_boxes[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(four_boxes[:, 0:4], 0, 2 * s, out=four_boxes[:, 0:4])  # use with random_affine



    # print("n1")
    # print(len(four_boxes))
    # print(four_boxes)
    # print(len(four_boxes[four_boxes[:,2]==0]))
    # print("n2")
    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    one_img, one_boxes = random_affine(four_img, four_boxes,
                                  # degrees=self.hyp['degrees'],
                                  # translate=self.hyp['translate'],
                                  # scale=self.hyp['scale'],
                                  # shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove




    # for box2 in four_boxes:
    #
    #     box = box2[0:4]
    #     box = box.astype(np.int32)
    #     print("sssssss",box)
    #     cv2.rectangle(four_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
    # cv2.imshow('c', four_img)
    # cv2.waitKey(0)
    return four_img, four_boxes,one_img, one_boxes



_data_rng = np.random.RandomState(None)

def np_random_color_distort(image, data_rng=None, eig_val=None,
                            eig_vec=None, var=0.4, alphastd=0.1):
    """Numpy version of random color jitter.
    Parameters
    ----------
    image : numpy.ndarray
        original image.
    data_rng : numpy.random.rng
        Numpy random number generator.
    eig_val : numpy.ndarray
        Eigen values.
    eig_vec : numpy.ndarray
        Eigen vectors.
    var : float
        Variance for the color jitters.
    alphastd : type
        Jitter for the brightness.
    Returns
    -------
    numpy.ndarray
        The jittered image
    """
    # from ....utils.filesystem import try_import_cv2
    # cv2 = try_import_cv2()
    if data_rng is None:
        data_rng = _data_rng
    if eig_val is None:
        eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                           dtype=np.float32)
    if eig_vec is None:
        eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                            [-0.5832747, 0.00994535, -0.81221408],
                            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def lighting_(data_rng, image, alphastd, eigval, eigvec):
        alpha = data_rng.normal(scale=alphastd, size=(3, ))
        image += np.dot(eigvec, eigval * alpha)

    def blend_(alpha, image1, image2):

        image1 *= alpha
        image2 *= (1 - alpha)



        image1 += image2


    def saturation_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        blend_(alpha, image, gs[:, :, None])

    def brightness_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image *= alpha


    def contrast_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        blend_(alpha, image, gs_mean)

    functions = [brightness_, contrast_, saturation_]
    np.random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    image = image.astype(np.float)
    gs = gs.astype(np.float)


    for f in functions:
        f(data_rng, image, gs, gs_mean, var)
    lighting_(data_rng, image, alphastd, eig_val, eig_vec)
    image = image.astype(np.uint8)
    return image

