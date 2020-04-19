"""Kernel density estimation"""

import numpy as np
import scipy.signal as sg


def kde2(points, grid_x, grid_y, bandwidth=0.2):
    """perform kernel density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width / 2)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width / 2)

    bins, edges = np.histogramdd(points, bins=(x_edges, y_edges))
    coeff = 1 / (np.sqrt(2 * np.pi) * bins.size * bandwidth)
    kernel = coeff * np.exp(-(grid_x ** 2 + np.vstack(grid_y) ** 2)/(2 * bandwidth ** 2))

    return sg.fftconvolve(bins.T, kernel, mode="same")


def dkde2(points, grid_x, grid_y, bandwidth=0.2):
    """perform derivative of density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width / 2)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width / 2)

    bins_, edges = np.histogramdd(points, bins=(x_edges, y_edges))
    grid_y = np.vstack(grid_y)

    exponential = np.exp(-(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))

    coeff = -1 / (np.sqrt(2 * np.pi) * bins_.size * bandwidth ** 2)
    k_dx = coeff * grid_x / bandwidth * exponential
    k_dy = coeff * grid_y / bandwidth * exponential

    coeff2 = 1 / (np.sqrt(2 * np.pi) * bins_.size * bandwidth ** 3)
    k_dxx = (grid_x ** 2 / bandwidth ** 2 - 1) * coeff2 * exponential
    k_dyy = (grid_y ** 2 / bandwidth ** 2 - 1) * coeff2 * exponential
    k_dxy = k_dyx = (grid_x * grid_y) / bandwidth ** 2 * coeff2 * exponential

    # k_dx = - grid_x / (2 * np.pi * bandwidth ** 4) * np.exp(
    #     -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    # k_dy = - grid_y / (2 * np.pi * bandwidth ** 4) * np.exp(
    #     -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    # k_dxx = (-1 + grid_x ** 2 / bandwidth ** 2) / (2 * np.pi * bandwidth ** 4) * np.exp(
    #     -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    # k_dyy = (-1 + grid_y ** 2 / bandwidth ** 2) / (2 * np.pi * bandwidth ** 4) * np.exp(
    #     -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))
    # k_dxy = k_dyx = (grid_x * grid_y) / (2 * np.pi * bandwidth ** 4) * np.exp(
    #     -(grid_x ** 2 + grid_y ** 2)/(2 * bandwidth ** 2))

    dx = sg.fftconvolve(bins_.T, k_dx, mode="same")
    dy = sg.fftconvolve(bins_.T, k_dy, mode="same")
    dxx = sg.fftconvolve(bins_.T, k_dxx, mode="same")
    dyy = sg.fftconvolve(bins_.T, k_dyy, mode="same")
    dxy = sg.fftconvolve(bins_.T, k_dxy, mode="same")
    dyx = sg.fftconvolve(bins_.T, k_dyx, mode="same")

    return dx, dy, dxx, dyy, dxy, dyx
