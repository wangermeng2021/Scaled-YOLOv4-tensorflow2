"""Extended image transformations to `mxnet.image`."""
from __future__ import division
import random
import numpy as np
import cv2

__all__ = ['imresize',
           'random_pca_lighting', 'random_expand', 'random_flip']

def imresize(src, w, h, inter=1):
    """Resize image with OpenCV.
    This is a duplicate of mxnet.image.imresize for name space consistency.
    Parameters
    ----------
    src : mxnet.nd.NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default='1'
        Interpolation method (default=cv2.INTER_LINEAR).
    out : NDArray, optional
        The output NDArray to hold the result.
    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    Examples
    --------
    """

    # oh, ow, _ = src.shape
    # return mx.image.imresize(src, w, h, interp=get_interp(interp, (oh, ow, h, w)))



    if inter == 0:
        inter_type = cv2.INTER_NEAREST
    elif inter == 1:
        inter_type = cv2.INTER_LINEAR
    elif inter == 2:
        inter_type = cv2.INTER_LINEAR_EXACT
    elif inter == 3:
        inter_type = cv2.INTER_AREA
    elif inter == 4:
        inter_type = cv2.INTER_CUBIC
    src = cv2.resize(src, (w,h), interpolation = inter_type)
    return src

def random_pca_lighting(src, alphastd, eigval=None, eigvec=None):
    """Apply random pca lighting noise to input image.
    Parameters
    ----------
    img : mxnet.nd.NDArray
        Input image with HWC format.
    alphastd : float
        Noise level [0, 1) for image with range [0, 255].
    eigval : list of floats.
        Eigen values, defaults to [55.46, 4.794, 1.148].
    eigvec : nested lists of floats
        Eigen vectors with shape (3, 3), defaults to
        [[-0.5675, 0.7192, 0.4009],
         [-0.5808, -0.0045, -0.8140],
         [-0.5836, -0.6948, 0.4203]].
    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    """
    if alphastd <= 0:
        return src

    if eigval is None:
        eigval = np.array([55.46, 4.794, 1.148])
    if eigvec is None:
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])

    alpha = np.random.normal(0, alphastd, size=(3,))
    rgb = np.dot(eigvec * alpha, eigval)
    src += nd.array(rgb, ctx=src.context)
    return src

def random_expand(src, max_ratio=4, fill=0, keep_ratio=True):
    """Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with HWC format.
    max_ratio : int or float
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill : int or float or array-like
        The value(s) for padded borders. If `fill` is numerical type, RGB channels
        will be padded with single value. Otherwise `fill` must have same length
        as image channels, which resulted in padding with per-channel values.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.
    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (offset_x, offset_y, new_width, new_height)
    """
    if max_ratio <= 1:
        return src, (0, 0, src.shape[1], src.shape[0])

    h, w, c = src.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=src.dtype)
    # make canvas
    # if isinstance(fill, np.numeric_types):
    #     dst = np.full(shape=(oh, ow, c), val=fill, dtype=src.dtype)
    # else:
    #     fill = np.array(fill, dtype=src.dtype)
    #     if not c == fill.size:
    #         raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
    #     dst = np.tile(fill.reshape((1, c)), reps=(oh * ow, 1)).reshape((oh, ow, c))

    dst[off_y:off_y+h, off_x:off_x+w, :] = src
    return dst, (off_x, off_y, ow, oh)

def random_flip(src, px=0, py=0, copy=False):
    """Randomly flip image along horizontal and vertical with probabilities.
    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image with HWC format.
    px : float
        Horizontal flip probability [0, 1].
    py : float
        Vertical flip probability [0, 1].
    copy : bool
        If `True`, return a copy of input
    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (flip_x, flip_y), records of whether flips are applied.
    """
    flip_y = np.random.choice([False, True], p=[1-py, py])
    flip_x = np.random.choice([False, True], p=[1-px, px])
    if flip_y:
        src = np.flip(src, axis=0)
    if flip_x:
        src = np.flip(src, axis=1)
    if copy:
        src = np.copy()
    return src, (flip_x, flip_y)
#