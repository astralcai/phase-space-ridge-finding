"""This file contains implementations of kernel density estimation"""

import numpy as np

from numba import jit


@jit(nopython=True)
def gaussian2d(x0, x, d, bw):
    """A vectorized 2D gaussian kernel function

    Args:
        x0: an array of source coordinates
        x: the target coordinates
        d: the dimension of the space
        bw: the bandwidth

    Returns:
        the gaussian function centered at x0, evaluated on x

    """

    coeff = 1 / np.sqrt(2 * np.pi) / bw
    result = 0
    for i in range(len(x0)):
        dist = np.sum((x - x0[i]) ** 2)
        result += coeff * np.exp(-dist / (2 * bw ** 2))
    return result


@jit(nopython=True)
def compute_kernel_sums(targets, sources, kernel, d, bw):
    """Computes the sum of all kernels on specified locations

    Args:
        targets (np.ndarray): an array of target grid points
        sources (np.ndarray): an array of source data points
        kernel (callable): the kernel function
        d (int): the number of dimensions (defaults to 2 right now)
        bw (float): the kernel bandwidth

    Returns:
        an array of density estimates for each target location

    """

    d = 2  # right now only 2 dimensions are supported

    pdf = np.zeros(targets.shape[0])
    for i in range(len(pdf)):
        pdf[i] = kernel(sources, targets[i], d, bw)
    return pdf


def kde2(targets, sources, d, bw):
    """Computes 2D KDE with an array of points"""

    pdf = compute_kernel_sums(targets, sources, gaussian2d, d, bw)
    return pdf / sources.shape[0]  # normalize
