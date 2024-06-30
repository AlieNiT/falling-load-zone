import numpy as np

def Z_to_XYZ(Z, f, w, h): # Z: (h, w) -> XYZ: (h, w, 3)
    y, x = np.mgrid[0:h, 0:w] - np.array([h/2-0.5, w/2-0.5]).reshape(2, 1, 1)
    X = x * Z / f
    Y = y * Z / f
    return np.stack([X, Y, Z], axis=-1)

def XYZ_to_xy(XYZ, f, w, h) -> np.ndarray: # XYZ: (..., 3) -> xy: (..., 2)
    x = XYZ[..., 0] * f / XYZ[..., 2] # (...)
    y = XYZ[..., 1] * f / XYZ[..., 2] # (...)
    x = x + w//2
    y = h//2 + y
    return np.stack([x, y], axis=-1).astype(np.int32)

def fit_plane_to_points(points):
    # points: n x 3
    centroid = points.mean(axis=0)
    points = points - centroid

    # Singular Value Decomposition
    _, _, V = np.linalg.svd(points)
    # every point on the plane can be represented as centroid + t * V[2] + s * V[1]
    return V[0], V[1], centroid

def project_points_to_plane(points, plane):
    V, U, centroid = plane
    shape = points.shape
    points = np.reshape(points, (-1, 3))
    t = np.sum((points - centroid) * V, axis=-1) / np.sum(V * V)
    s = np.sum((points - centroid) * U, axis=-1) / np.sum(U * U)
    projected_points = centroid + t[:, None] * V + s[:, None] * U
    return np.reshape(projected_points, shape)