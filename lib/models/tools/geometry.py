# modified from https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
import numpy as np
import torch
import transforms3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def fit_plane(points):
    """
    Fit a plane from points

    Args:
        points: [N, 3]
    """

    p_mean = points.mean(0)
    p_centered = points - p_mean[None]

    U, S, Vh = np.linalg.svd(p_centered)
    normal = Vh[2, :]
    d = -np.dot(normal, p_mean)

    return normal, d, p_centered


def rodrigues_rot(p, n0, n1):
    if p.ndim == 1:
        p = p[np.newaxis]

    # Get vector of rotation axis k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)

    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)

    theta = np.arccos(np.dot(n0, n1))

    # compute rotated points
    rot_mat = transforms3d.axangles.axangle2mat(k, theta, is_normalized=True)

    p_rot = p @ rot_mat.T

    return p_rot


def fit_circle_2d(x, y, w=[]):
    # -------------------------------------------------------------------------------
    # FIT CIRCLE 2D
    # - Find center [xc, yc] and radius r of circle fitting to set of 2D points
    # - Optionally specify weights for points
    #
    # - Implicit circle function:
    #   (x-xc)^2 + (y-yc)^2 = r^2
    #   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
    #   c[0]*x + c[1]*y + c[2] = x^2+y^2
    #
    # - Solution by method of least squares:
    #   A*c = b, c' = argmin(||A*c - b||^2)
    #   A = [x y 1], b = [x^2+y^2]
    # -------------------------------------------------------------------------------
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r


def angle_between(u, v, n=None):
    # -------------------------------------------------------------------------------
    # ANGLE BETWEEN
    # - Get angle between vectors u,v with sign based on plane with unit normal n
    # -------------------------------------------------------------------------------
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))


def fit_circle_3d(points):
    # 1. fit a 2d plane of the points

    n, d, p_centered = fit_plane(points=points)

    # 2. project points to coords X-Y in the 2d plane
    p_xy = rodrigues_rot(p_centered, n, [0, 0, 1])

    # 3. fit a circle on the plane
    xc, yc, r = fit_circle_2d(p_xy[:, 0], p_xy[:, 1])

    # transform center back to 3D coordinates

    p_mean = points.mean(0)

    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], n) + p_mean
    C = C.flatten()

    return C, n, r, d


def generate_circle_by_vectors(t, C, r, n, u):
    # Generate points on circle
    # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    """
    Args:
        t: [0, 2π)
        C: center of the circle
        r: radius of the circle
        n: normal of the circle plane
        u: a vector on the circle plane
    """

    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    p_c = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * (np.cross(n, u)) + C[np.newaxis]

    return p_c


def generate_circle_by_angles(t, C, r, theta, phi):
    # orthonormal vectors n, u, <n, u> = 0
    n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    u = np.array([-np.sin(phi), np.cos(phi), 0])

    p_c = generate_circle_by_vectors(t, C, r, n, u)

    return p_c


def center_radius_from_poses(poses):
    ''' Get the center poisition and radius supposing all cameras looking at the same point '''
    rays_o = poses[:, :3, 3:4]
    rays_d = poses[:, :3, 2:3]
    def min_line_dist(rays_o, rays_d):
        if isinstance(rays_o, np.ndarray) and isinstance(rays_d, np.ndarray):
            A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -A_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        else:
            A_i = rays_o.new_ones((3, )) - rays_d * rays_d.permute(0, 2, 1)
            b_i = -A_i @ rays_o
            pt_mindist = torch.squeeze(-torch.linalg.inv((torch.transpose(A_i, 1, 2) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist
    
    center_pt = min_line_dist(rays_o, rays_d)
    rays_o = rays_o - center_pt[None]
    
    if isinstance(rays_o, np.ndarray):
        radius = np.mean(np.linalg.norm(rays_o, axis=1))
    else:
        radius = torch.mean(torch.linalg.norm(rays_o, dim=1))
    
    return center_pt, radius

def point_line_distance(p, rays_o, rays_d):
    # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    if p.ndim == 1:
        p = p[None]
    
    rays_d = rays_d / torch.linalg.norm(rays_d)
    d = torch.sum(torch.cross((p - rays_o), rays_d) * torch.cross((p - rays_o), rays_d), dim=1)
    
    return d


def point_plane_distance(p, normal, d):
    
    if p.ndim == 1:
        p = p[None]
    
    if normal.ndim == 1:
        normal = normal[None]
    
    if d.ndim == 1:
        d = d[None]

    distance = torch.abs((normal @ p.T + d) / torch.linalg.norm(normal, 1))
    
    return distance