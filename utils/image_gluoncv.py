"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import random
import cv2

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
        image = image.astype(np.float32) + np.dot(eigvec, eigval * alpha)

        image = image.astype(np.uint8)
        return image
    def blend_(alpha, image1, image2):

        image1 = image1.astype(np.float32)*alpha
        image2 = image2.astype(np.float32)*(1 - alpha)
        image1 += image2
        image1 = image1.astype(np.uint8)
        return image1
    def saturation_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        return blend_(alpha, image, gs[:, :, None])

    def brightness_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        # print(alpha)
        # image *= alpha
        image = image.astype(np.float32)*alpha
        image = image.astype(np.uint8)
        return image
    def contrast_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        return blend_(alpha, image, gs_mean)

    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        image = f(data_rng, image, gs, gs_mean, var)
    image = lighting_(data_rng, image, alphastd, eig_val, eig_vec)
    # print(data_rng)
    return image