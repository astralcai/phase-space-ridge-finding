"""Kernel density estimation"""

import numpy as np
import scipy.signal as sg


def kde2(points, grid_x, grid_y, bandwidth=0.2):
    """perform kernel density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width)

    bins, edges = np.histogramdd(points, bins=(x_edges, y_edges), normed=True)
    kernel = 1 / (2 * np.pi * bandwidth ** 2) * np.exp(
        -(grid_x ** 2 + np.vstack(grid_y) ** 2)/(2 * bandwidth ** 2))

    return sg.fftconvolve(bins, kernel, mode="same")


def dkde2(points, grid_x, grid_y, bins=128, bandwidth=0.2):
    """perform derivative of density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width)

    bins, edges = np.histogramdd(points, bins=(x_edges, y_edges), normed=True)
    grid_y = np.vstack(grid_y)

    k_dx = - grid_x / (2 * np.pi * bandwidth ** 4) * np.exp(
        -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    k_dy = - grid_y / (2 * np.pi * bandwidth ** 4) * np.exp(
        -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    k_dxx = (-1 + grid_x ** 2 / bandwidth ** 2) / (2 * np.pi * bandwidth ** 4) * np.exp(
        -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    k_dyy = (-1 + grid_y ** 2 / bandwidth ** 2) / (2 * np.pi * bandwidth ** 4) * np.exp(
        -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    k_dxy = k_dyx = (grid_x * grid_y) / (2 * np.pi * bandwidth ** 6) * np.exp(
        -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))

    dx = sg.fftconvolve(bins, k_dx, mode="same")
    dy = sg.fftconvolve(bins, k_dy, mode="same")
    dxx = sg.fftconvolve(bins, k_dxx, mode="same")
    dyy = sg.fftconvolve(bins, k_dyy, mode="same")
    dxy = sg.fftconvolve(bins, k_dxy, mode="same")
    dyx = sg.fftconvolve(bins, k_dyx, mode="same")

    return dx, dy, dxx, dyy, dxy, dyx
