"""Kernel density estimation"""

import numpy as np
import scipy.signal as sg


def kde2(points, grid_x, grid_y, bandwidth=0.2):
    """perform kernel density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width)

    bins, edges = np.histogramdd(points, bins=(x_edges, y_edges), density=True)
    kernel = 1 / (2 * np.pi * bandwidth ** 2) * np.exp(
        -(grid_x ** 2 + np.vstack(grid_y) ** 2)/(2 * bandwidth ** 2))

    return sg.fftconvolve(bins.T, kernel, mode="same")


def dkde2(points, grid_x, grid_y, bins=128, bandwidth=0.2):
    """perform derivative of density estimation with gaussian kernel"""

    x_bin_width, y_bin_width = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    x_edges = np.append(grid_x - x_bin_width / 2, grid_x[-1] + x_bin_width)
    y_edges = np.append(grid_y - y_bin_width / 2, grid_y[-1] + y_bin_width)

    bins_, edges = np.histogramdd(
        points, bins=(x_edges, y_edges), density=True)
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

    dx = sg.fftconvolve(bins_.T, k_dx, mode="same")
    dy = sg.fftconvolve(bins_.T, k_dy, mode="same")
    dxx = sg.fftconvolve(bins_.T, k_dxx, mode="same")
    dyy = sg.fftconvolve(bins_.T, k_dyy, mode="same")
    dxy = sg.fftconvolve(bins_.T, k_dxy, mode="same")
    dyx = sg.fftconvolve(bins_.T, k_dyx, mode="same")

    return dx, dy, dxx, dyy, dxy, dyx


def kden(points, bw, *grids):
    """perform n-dimensional KDE"""

    d = points.shape[1]  # dimensions

    grids = np.asarray(grids)
    bin_widths = grids[:, 1] - grids[:, 0]
    edges = np.append(grids - bin_widths / 2, grids[:, -1] + bin_widths)

    grid_meshes = np.meshgrid(*grids, indexing='xy')

    bins, _ = np.histogramdd(points, bins=edges, density=True)
    kernel = 1 / (2 * np.pi) ** (d / 2) / bw ** d * np.exp(
        -np.sum(grid_meshes ** 2, axis=0) / (2 * bw ** 2))

    return sg.fftconvolve(bins.T, kernel, mode="same")


def dkden(points, bw, *grids):
    """perform n-dimensional KDE derivatives"""

    d = points.shape[1]  # dimensions

    grids = np.asarray(grids)
    bin_widths = grids[:, 1] - grids[:, 0]
    edges = np.append(grids - bin_widths / 2, grids[:, -1] + bin_widths)

    grid_meshes = np.meshgrid(*grids, indexing='xy')
    bins, _ = np.histogramdd(points, bins=edges, density=True)
