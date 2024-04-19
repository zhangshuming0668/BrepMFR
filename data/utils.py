import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def bounding_box_uvgrid(inp_n: torch.Tensor, inp_e: torch.Tensor):
    pts_n = inp_n[..., :3].reshape((-1, 3))
    mask = inp_n[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts_n = pts_n[point_indices_inside_faces, :]
    pts_e = inp_e[..., :3].reshape((-1, 3))
    pts = torch.cat([pts_n, pts_e], 0)
      
    return bounding_box_pointcloud(pts)


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp_n: torch.Tensor, inp_e: torch.Tensor):
    bbox = bounding_box_uvgrid(inp_n, inp_e)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])

    return center, scale


def center_and_scale_pointcloud(pnts: torch.Tensor):
    bbox = bounding_box_pointcloud(pnts)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    return center, scale


def get_random_rotation():
    """Get a random rotation in 90 degree increments along the canonical axes"""
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """Rotate the node features in the graph by a given rotation"""
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(
        orig_size
    )  # Normals/tangents
    return inp
