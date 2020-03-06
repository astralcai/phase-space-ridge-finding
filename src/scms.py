"""Subspace constrained mean shift algorithm"""

import numpy as np
import matplotlib.pyplot as plt

def normalize(vec):
    norm = np.linalg.norm(vec)
    if (norm == 0):
        return vec
    return vec/norm


def generate_mesh2(x, y, threshold, pdf):
    """Generate the original 2D mesh of points"""

    # create original mesh
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # thresholding
    xx_ma = np.ma.masked_where(pdf < threshold, xx)
    yy_ma = np.ma.masked_where(pdf < threshold, yy)

    # convert meshgrid to an array of points
    points = np.array([xx_ma.compressed().ravel(), yy_ma.compressed().ravel()]).T

    return points


def interpolate2(x, y, X, Y, vals):
    """interpolate a value for a given point in 2D
    
    TODO: try derivative based interpolation:
    https://dspace.library.uu.nl/bitstream/handle/1874/580/c4.pdf
    
    """

    if x < X[0] or x > X[-1] or y < Y[0] or y > Y[-1]:
        return 0

    dx, dy = X[1] - X[0], Y[1] - Y[0]

    i = int((x - X[0]) // dx)
    j = int((y - Y[0]) // dy)

    v11, v12, v21, v22 = vals[i, j], vals[i, j+1], vals[i+1, j], vals[i+1, j+1]

    result = 1 / (dx * dy) * np.dot(np.dot(
        np.array([X[i + 1] - x, x - X[i]]), np.array(
            [[v11, v12], [v21, v22]])), np.vstack([Y[j + 1] - y, y - Y[j]]))

    return result[0]


def get_derivatives2(x, y, X, Y, pdf, dpdf, ddpdf):
    """Calculate first and second order derivatives for a given point in 2D"""

    DX, DY = dpdf
    DXX, DYY, DXY, DYX = ddpdf

    dx = interpolate2(x, y, X, Y, DX)
    dy = interpolate2(x, y, X, Y, DY)
    dxx = interpolate2(x, y, X, Y, DXX)
    dyy = interpolate2(x, y, X, Y, DYY)
    dxy = interpolate2(x, y, X, Y, DXY)
    dyx = interpolate2(x, y, X, Y, DYX)

    return dx, dy, dxx, dyy, dxy, dyx


def meanshift(point, x, y, pdf, dpdf, ddpdf):
    """Calculates the mean shift vector"""

    dx, dy, dxx, dyy, dxy, dyx = get_derivatives2(
        point[0], point[1], x, y, pdf, dpdf, ddpdf)
    Hessian = np.array([[dxx, dxy], [dyx, dyy]])
    e_vals, e_vecs = np.linalg.eigh(Hessian)
    idx = np.argmin(e_vals)
    V = np.delete(e_vecs, idx, 0)
    VM = np.matmul(V.T, V)
    MS = np.array([dx, dy])
    MC = np.matmul(VM, MS)
    return normalize(MC)


def rk_approximate(point, x, y, pdf, dpdf, ddpdf, h):
    px, py = point
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    # First Step
    k1 = h * MC
    # Step 2
    px, py = point + k1 / 5
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    k2 = h * MC
    #Step 3
    px, py = point + k1 * 3. / 40 + k2 * 9. / 40
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    k3 = h * MC
    #Step 4
    px, py = point + k1 * 3. / 10 + k2 * (-9./10) + k3 * 6. / 5
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    k4 = h * MC
    #Step 5
    px, py = point+k1*(-11./54)+k2*5./2+k3*(-70/27)+k4*35./27
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    k5 = h * MC
    #Step 6
    px, py = point+k1*(1631./55296)+k2*(175./512)+k3*(575./13824)+k4*(44275./110592)+k5*(253./4096)
    MC = meanshift([px, py], x, y, pdf, dpdf, ddpdf)
    k6 = h * MC
    #Fifth-order Runge-Kutta formula
    result = point+k1*37/378+k3*250./621+k4*125./594+k6*512./1771
    
    #Embedded fourth-order Runge-Kutta formula
    result2 = point+k1*2825./27648+k3*18575./48384+k4*13525./55296+k5*277./14336+k6*1./4

    return result, result2


def scms2(points, x, y, pdf, dpdf, ddpdf, iterations=20):
    """Perform 2D subspacee constrained mean shift algorithm on the given points"""

    dx0 = 0.000001
    dy0 = 0.000001
    momentum = np.zeros((len(points), 2))
    H = np.zeros(len(points)) + 0.05  # step sizes
    for i in range(iterations):
        for i in range(len(points)):
            pt1, pt2 = rk_approximate(points[i], x, y, pdf, dpdf, ddpdf, H[i])
            dx1 = abs(pt2[0]-pt1[0])
            modx = 2 if dx1 == 0 else min(2, (dx0 / dx1) ** 0.2)
            dy1 = abs(pt2[1]-pt1[1])
            mody = 2 if dy1 == 0 else min(2, (dy0 / dy1) ** 0.2)
            mod = min(modx, mody)
            if(mod < 1): # d1 > d0, we have to retry! with new step
                H[i] = H[i] * mod
                pt1, pt2 = rk_approximate(points[i], x, y, pdf, dpdf, ddpdf, H[i])
                new_pt = pt2 + momentum[i] * 0.5
                momentum[i] = pt2 - points[i]
                points[i] = new_pt
            else: # we can safely increase the next step size
                H[i] = H[i] * mod
                new_pt = pt2 + momentum[i] * 0.5
                momentum[i] = pt2 - points[i]
                points[i] = new_pt
        # plt.plot(points[:,0], points[:,1], 'k.', markersize=0.3)
        # plt.show()

    return points