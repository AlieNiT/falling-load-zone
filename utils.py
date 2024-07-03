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


def distance_to_plane(XYZ, plane):
    v1, v2, c = plane
    normal_vector = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal_vector)
    normal_vector_normalized = normal_vector / normal_norm
    c = c.reshape(1, 1, 3)
    normal_vector_normalized = normal_vector_normalized.reshape(1, 1, 3)
    diff = XYZ - c
    dot_product = np.sum(diff * normal_vector_normalized, axis=2)
    distances = np.abs(dot_product)
    return distances

def fit_ground_single(XYZ, p):
    px, py = p
    candidates = XYZ[py-20:py+20, px-50:px+50].reshape((-1, 3))
    for i in range(5):
        # print(candidates.shape)
        if len(candidates) > 5000:
            candidates = candidates[np.random.choice(len(candidates), 5000, replace=False)]
        plane = fit_plane_to_points(candidates)
        # print(plane)
        dist = distance_to_plane(XYZ, plane)
        dist_filter = dist < 0.1 / 2**i
        candidates = XYZ[dist_filter]
    return plane


def fit_ground(XYZ):
    height = XYZ.shape[0]
    width = XYZ.shape[1]
    p1x = int(width * 1/6)
    p2x = int(width * 4/6)
    p3x = int(width * 5/6)
    p1y = int(height-20)
    p2y = int(height-20)
    p3y = int(height-20)
    
    ps = [(p1x, p1y), (p2x, p2y), (p3x, p3y)]
    planes = []
    xs = []

    #return fit_ground_single(XYZ, ps[0])

    for p in ps:
        plane = fit_ground_single(XYZ, p)
        planes.append(plane)
        v1, v2, c = plane
        x = np.cross(v1, v2)
        x = x / np.linalg.norm(x)
        xs.append(x)

    # majority vote
    x01 = abs(np.dot(xs[0], xs[1]))
    x02 = abs(np.dot(xs[0], xs[2]))
    x12 = abs(np.dot(xs[1], xs[2]))
    xxx = min(x01, x02, x12)
    if xxx == x01:
        return planes[0]
    if xxx == x02:
        return planes[2]
    if xxx == x12:
        return planes[1]

def IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union