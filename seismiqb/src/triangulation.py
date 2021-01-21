""" Triangulation functions. """
import numpy as np
from numba import njit

@njit
def triangle_rasterization(points, width=1):
    """ Transform triangle to surface of the fixed thickness.

    Parameters
    ----------
    points : numpy.ndarray
        array of size 3 x 3: each row is a vertex of triangle
    width : int
        thicc

    Return
    ------
    numpy.ndarray
        array of size N x 3 where N is a number of points in rasterization.
    """
    shape = np.array([np.max(points[:, i]) - np.min(points[:, i]) for i in range(3)])
    _points = np.zeros((int((shape[0] + 2) * (shape[1] + 2) * (shape[2] + 2)), 3))
    i = 0
    for x in range(int(np.min(points[:, 0])), int(np.max(points[:, 0])+1)): # pylint: disable=not-an-iterable
        for y in range(int(np.min(points[:, 1])), int(np.max(points[:, 1]+1))):
            for z in range(int(np.min(points[:, 2])), int(np.max(points[:, 2]+1))):
                node = np.array([x, y, z])
                if distance_to_triangle(points, node) < 0.5 * width:
                    _points[i] = node
                    i += 1
    return _points[:i]

def make_triangulation(points, return_indices=False):
    """ Compute triangulation of the fault.

    Parameters
    ----------
    points : numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    return_indices : bool
        If True, function will return indices of stick nodes in flatten array.

    Return
    ------
    numpy.ndarray
        numpy.ndarray of length N where N is the number of simplices. Each item is a sequence of coordinates of each
        vertex (if `return_indices=False`) or indices of nodes in initial flatten array.
    """
    triangles = []
    if return_indices:
        n_points = np.cumsum([0, *[len(item) for item in points]])
        points = np.array([np.arange(len(points[i])) + n_points[i] for i in range(len(points))])
    for s1, s2 in zip(points[:-1], points[1:]):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        n = len(s1)
        nodes = [item for sublist in zip(s1, s2[:n]) for item in sublist]
        nodes = [nodes[i:i+3] for i in range(len(nodes[:-2]))] if len(nodes) > 2 else []
        nodes += [[s1[-1], s2[i], s2[i+1]] for i in range(n-1, len(s2)-1)]
        triangles += nodes
    return np.array(triangles)


@njit
def distance_to_triangle(triangle, node):
    """ https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e """
    # pylint: disable=invalid-name, too-many-nested-blocks, too-many-branches, too-many-statements
    B = triangle[0, :]
    E0 = triangle[1, :] - B
    E1 = triangle[2, :] - B
    D = B - node
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0
    dist = np.sqrt(sqrdistance)
    return dist
