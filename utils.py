import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Union
from torch_scatter import scatter_min
from tqdm import tqdm
from color_utils import histogram, histogram_intersection
from collections import defaultdict
from math import ceil, sqrt
import matplotlib.pyplot as plt
from typing import Optional
import math


def cloud2idx(xyz: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """
    Change 3d coordinates to image coordinates ranged in [-1, 1].

    Args:
        xyz: (N, 3) torch tensor containing xyz values of the point cloud data
        batched: If True, performs batched operation with xyz considered as shape (B, N, 3)

    Returns:
        coord_arr: (N, 2) torch tensor containing transformed image coordinates
    """
    if batched:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[..., :2], dim=-1)), xyz[..., 2] + 1e-6), -1)  # (B, N, 1)

        # horizontal angle
        phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1] + 1e-6)  # (B, N, 1)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)  # (B, N, 2)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[..., 0] / (np.pi * 2), sphere_cloud_arr[..., 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)  # (B, N, 2)

    else:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

        # horizontal angle
        phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)

    return coord_arr


def sample_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear', batched=False) -> torch.Tensor:
    """
    Image sampling function
    Use coord_arr as a grid for sampling from img

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        coord_arr: (N, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid
        batched: If True, assumes an additional batch dimension for coord_arr

    Returns:
        sample_rgb: (N, 3) torch tensor containing sampled RGB values
    """
    if batched:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(coord_arr.shape[0], coord_arr.shape[1], 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img.expand(coord_arr.shape[0], -1, -1, -1), sample_arr, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(sample_rgb)  # (B, 3, N)
        sample_rgb = torch.transpose(sample_rgb, 1, 2)  # (B, N, 3)  

    else:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(1, -1, 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img, sample_arr, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(torch.squeeze(sample_rgb, 0), 2)
        sample_rgb = torch.transpose(sample_rgb, 0, 1)

    return sample_rgb


def warp_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear') -> torch.Tensor:
    """
    Image warping function
    Use coord_arr as a grid for warping from img

    Args:
        img: (H, W, C) torch tensor containing image RGB values
        coord_arr: (H, W, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid

    Returns:
        sample_rgb: (H, W, C) torch tensor containing sampled RGB values
    """

    img = img.permute(2, 0, 1)  # (C, H, W)
    img = torch.unsqueeze(img, 0)  # (1, C, H, W)

    # sampling from img
    sample_arr = coord_arr.unsqueeze(0)  # (1, H, W, 2)
    sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
    sample_rgb = F.grid_sample(img, sample_arr, align_corners=False, padding_mode=padding, mode=mode)  # (1, C, H, W)

    sample_rgb = sample_rgb.squeeze(0).permute(1, 2, 0)  # (H, W, C)

    return sample_rgb


def make_pano(xyz: torch.Tensor, rgb: torch.Tensor, resolution: Tuple[int, int] = (200, 400), return_torch: bool = False) -> Union[torch.Tensor, np.array]:
    """
    Make panorama image from xyz and rgb tensors

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        rgb: (N, 3) torch tensor containing rgb values, ranged in [0, 1]
        resolution: Tuple size of 2, returning panorama image of size resolution
        return_torch: if True, return image as torch.Tensor
                      if False, return image as numpy.array

    Returns:
        image: (H, W, 3) torch.Tensor or numpy.array
    """

    with torch.no_grad():

        # project farther points first
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        mod_rgb = rgb.clone().detach()[mod_idx]

        coord_idx = cloud2idx(mod_xyz)
        coord_idx = (coord_idx + 1.0) / 2.0
        # coord_idx[:, 0] is x coordinate, coord_idx[:, 1] is y coordinate
        coord_idx[:, 0] *= (resolution[1] - 1)
        coord_idx[:, 1] *= (resolution[0] - 1)

        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = coord_idx.long()
        coord_idx = tuple(coord_idx.t())

        image = torch.zeros([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)

        # color the image
        # pad by 1
        temp = torch.ones_like(coord_idx[0], device=xyz.device)
        coord_idx1 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx2 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      coord_idx[1])
        coord_idx3 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx4 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx5 = (torch.clamp(coord_idx[0] - temp, min=0),
                      coord_idx[1])
        coord_idx6 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx7 = (coord_idx[0],
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx8 = (coord_idx[0],
                      torch.clamp(coord_idx[1] - temp, min=0))

        image.index_put_(coord_idx8, mod_rgb, accumulate=False)
        image.index_put_(coord_idx7, mod_rgb, accumulate=False)
        image.index_put_(coord_idx6, mod_rgb, accumulate=False)
        image.index_put_(coord_idx5, mod_rgb, accumulate=False)
        image.index_put_(coord_idx4, mod_rgb, accumulate=False)
        image.index_put_(coord_idx3, mod_rgb, accumulate=False)
        image.index_put_(coord_idx2, mod_rgb, accumulate=False)
        image.index_put_(coord_idx1, mod_rgb, accumulate=False)
        image.index_put_(coord_idx, mod_rgb, accumulate=False)

        image = image * 255

        if not return_torch:
            image = image.cpu().numpy().astype(np.uint8)

    return image


def quantile(x: torch.Tensor, q: float) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Obtain q quantile value and (1 - q) quantile value from x

    Args:
        x: 1-dim torch tensor
        q: q value for quantile

    Returns:
        result_1: q quantile value of x
        result_2: (1 - q) quantile value of x
    """

    with torch.no_grad():
        inds = torch.argsort(x)
        val_1 = int(len(x) * q)
        val_2 = int(len(x) * (1 - q))

        result_1 = x[inds[val_1]]
        result_2 = x[inds[val_2]]

    return result_1, result_2


def out_of_room(xyz: torch.Tensor, trans: torch.Tensor, out_quantile: float = 0.05) -> bool:
    """
    Check if translation is out of xyz coordinates

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        trans: (3, 1) torch tensor containing xyz translation

    Returns:
        False if translation is not out of room
        True if translation is out of room
    """

    with torch.no_grad():
        # rejecting outliers
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

        if x_min < trans[0][0] < x_max and y_min < trans[1][0] < y_max and z_min < trans[2][0] < z_max:
            return False
        else:
            return True


def get_bound(xyz: torch.Tensor, cfg, return_brute=False):
    # Obtain bounds for use in bayesian optimization
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    with torch.no_grad():
        # rejecting outliers
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 *  np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    if return_brute:
        return (slice(x_min.item(), x_max.item()), slice(y_min.item(), y_max.item()), slice(z_min.item(), z_max.item()),
            slice(min_yaw, max_yaw), slice(min_pitch, max_pitch), slice(min_roll, max_roll))
    else:
        return {'x': (x_min.item(), x_max.item()), 'y': (y_min.item(), y_max.item()), 'z': (z_min.item(), z_max.item()),
            'yaw': (min_yaw, max_yaw), 'pitch': (min_pitch, max_pitch), 'roll': (min_roll, max_roll)}


def adaptive_trans_num(xyz: torch.Tensor, max_trans_num: int, xy_only: bool = False) -> Tuple[int, int]:
    """
    Make the number of translation x, y coordinate candidates

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates of point cloud data
        max_trans_num: maximum number of translation candidates
        xy_only: If True, initialize only on x, y

    Returns:
        num_start_trans_x: number of x coordinate translation candidates
        num_start_trans_y: number of y coordinate translation candidates
        num_start_trans_z: number of z coordinate translation candidates, only returned when xy_only is False
    """

    xyz_max = xyz.max(dim=0)[0]
    xyz_min = xyz.min(dim=0)[0]
    xyz_length = xyz_max - xyz_min

    if xy_only:
        num_start_trans_x = ceil((xyz_length[0] * max_trans_num / xyz_length[1]) ** (1 / 2))
        num_start_trans_y = ceil((xyz_length[1] * max_trans_num / xyz_length[0]) ** (1 / 2))

        return num_start_trans_x, num_start_trans_y
    else:
        num_start_trans_x = ceil((xyz_length[0] ** 2 * max_trans_num / (xyz_length[1] * xyz_length[2])) ** (1 / 3))
        num_start_trans_y = ceil((xyz_length[1] ** 2 * max_trans_num / (xyz_length[0] * xyz_length[2])) ** (1 / 3))
        num_start_trans_z = ceil((xyz_length[2] ** 2 * max_trans_num / (xyz_length[0] * xyz_length[1])) ** (1 / 3))

        if num_start_trans_x % 2 == 0:
            num_start_trans_x -= 1
        if num_start_trans_y % 2 == 0:
            num_start_trans_y -= 1
        if num_start_trans_z % 2 == 0:
            num_start_trans_z -= 1

        return num_start_trans_x, num_start_trans_y, num_start_trans_z


def generate_rot_points(init_dict=None, device='cpu'):
    """
    Generate rotation starting points

    Args:
        init_dict: Dictionary containing details of initialization
        device: Device in which rotation starting points will be saved

    Returns:
        rot_arr: (N, 3) array containing (yaw, pitch, roll) starting points
    """

    if init_dict['yaw_only']:
        rot_arr = torch.zeros(init_dict['num_yaw'], 3, device=device)
        rot = torch.arange(init_dict['num_yaw'], dtype=torch.float, device=device)
        rot = rot * 2 * np.pi / init_dict['num_yaw']
        rot_arr[:, 0] = rot

    else:
        # Perform 3 DoF initialization
        rot_coords = torch.meshgrid(torch.arange(init_dict['num_yaw'], device=device).float() / init_dict['num_yaw'],
            torch.arange(init_dict['num_pitch'], device=device).float() / init_dict['num_pitch'],
            torch.arange(init_dict['num_roll'], device=device).float() / init_dict['num_roll'])

        rot_arr = torch.stack([rot_coords[0].reshape(-1), rot_coords[1].reshape(-1), rot_coords[2].reshape(-1)], dim=0).t()

        rot_arr[:, 0] = (rot_arr[:, 0] * (init_dict['max_yaw'] - init_dict['min_yaw'])) + init_dict['min_yaw']
        rot_arr[:, 1] = (rot_arr[:, 1] * (init_dict['max_pitch'] - init_dict['min_pitch'])) + init_dict['min_pitch']
        rot_arr[:, 2] = (rot_arr[:, 2] * (init_dict['max_roll'] - init_dict['min_roll'])) + init_dict['min_roll']

        # Initialize grid sample locations
        grid_list = [compute_sampling_grid(ypr, init_dict['num_yaw'], init_dict['num_pitch']) for ypr in rot_arr]

        # Filter out overlapping rotations
        round_digit = 3
        rot_list = [str(np.around(grid.cpu().numpy(), round_digit)) for grid in grid_list]
        valid_rot_idx = [rot_list.index(rot_mtx) for rot_mtx in set(rot_list)]
        rot_arr = torch.stack([rot_arr[idx] for idx in valid_rot_idx], dim=0)

    return rot_arr


def generate_trans_points(xyz, init_dict=None, device='cpu'):
    """
    Generate translation starting points

    Args:
        xyz: Point cloud coordinates
        init_dict: Dictionary containing details of initialization
        device: Device in which translation starting points will be saved

    Returns:
        trans_arr: (N, 3) array containing (x, y, z) starting points
    """
    def get_starting_points(num_trans_x, num_trans_y, num_trans_z=None):
        if init_dict['trans_init_mode'] == 'uniform':
            x_points = (torch.arange(num_trans_x, device=device) + 1) / (num_trans_x + 1) * (xyz[:, 0].max() - xyz[:, 0].min()) + xyz[:, 0].min()
            y_points = (torch.arange(num_trans_y, device=device) + 1) / (num_trans_y + 1) * (xyz[:, 1].max() - xyz[:, 1].min()) + xyz[:, 1].min()
            if num_trans_z is not None:
                z_points = (torch.arange(num_trans_z, device=device) + 1) / (num_trans_z + 1) * (xyz[:, 2].max() - xyz[:, 2].min()) + xyz[:, 2].min()

        elif init_dict['trans_init_mode'] == 'quantile':
            x_points = torch.quantile(xyz[:, 0],
                                      (torch.arange(num_trans_x, device=device) + 1) / (num_trans_x + 1))
            y_points = torch.quantile(xyz[:, 1],
                                      (torch.arange(num_trans_y, device=device) + 1) / (num_trans_y + 1))
            if num_trans_z is not None:
                z_points = torch.quantile(xyz[:, 2], (torch.arange(num_trans_z, device=device) + 1) / (num_trans_z + 1))

        elif init_dict['trans_init_mode'] == 'manual':
            x_points = (torch.arange(num_trans_x, device=device)) / (num_trans_x - 1) * (init_dict['x_max'] - init_dict['x_min']) + init_dict['x_min']
            y_points = (torch.arange(num_trans_y, device=device)) / (num_trans_y - 1) * (init_dict['y_max'] - init_dict['y_min']) + init_dict['y_min']
            if num_trans_z is not None:
                z_points = (torch.arange(num_trans_z, device=device)) / (num_trans_z - 1) * (init_dict['z_max'] - init_dict['z_min']) + init_dict['z_min']
        if num_trans_z is not None:
            return x_points, y_points, z_points
        else:
            return x_points, y_points

    if init_dict['xy_only']:
        if init_dict['dataset'] == 'Stanford2D-3D-S' or init_dict['dataset'] == 'Matterport3D' or init_dict['dataset'] == 'hoam':
            num_trans_x, num_trans_y = adaptive_trans_num(xyz, init_dict['num_trans'], xy_only=True)
            trans_arr = torch.zeros(num_trans_x * num_trans_y, 3, device=device)

            x_points, y_points = get_starting_points(num_trans_x, num_trans_y)
            trans_coords = torch.meshgrid(x_points, y_points)
            trans_arr[:, :2] = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1)], dim=0).t()
            trans_arr[:, 2] = xyz[:, 2].mean()

        elif init_dict['dataset'] == 'MPO':
            num_trans_x, num_trans_y = adaptive_trans_num(xyz, init_dict['num_trans'], xy_only=True)
            num_trans_z = 3
            trans_arr = torch.zeros((num_trans_x * num_trans_y * num_trans_z, 3), dtype=torch.float, device=device)

            x_points, y_points = get_starting_points(num_trans_x, num_trans_y)
            z_min, z_max = quantile(xyz[:, 2], 0.05)
            z_mean = xyz[:, 2].mean()
            z_3 = torch.tensor([z_mean + (z_max - z_mean) / 2, z_mean, z_mean - (z_mean - z_min) / 2], dtype=torch.float, device=device)

            for i in range(num_trans_x * num_trans_y * num_trans_z):
                trans_arr[i] = torch.tensor([x_points[i // (num_trans_y * num_trans_z)],
                                        y_points[(i % (num_trans_y * num_trans_z)) // num_trans_z],
                                        z_3[(i % (num_trans_y * num_trans_z)) % num_trans_z]], dtype=torch.float, device=device)

        elif init_dict['dataset'] == 'Data61/2D3D':
            num_trans_x, num_trans_y = adaptive_trans_num(xyz, init_dict['num_trans'], xy_only=True)
            trans_arr = torch.zeros((num_trans_x * num_trans_y, 3), dtype=torch.float)

            x_points, y_points = get_starting_points(num_trans_x, num_trans_y)
            z_val = xyz[:, 2].median() + 1.7

            for i in range(num_trans_x * num_trans_y):
                trans_arr[i] = torch.tensor([x_points[i % num_trans_x], y_points[i // num_trans_x], z_val],
                                        dtype=torch.float, device=device)

    else:
        if init_dict['trans_init_mode'] == 'octree':
            trans_arr = generate_octree(xyz, device)
        else:
            num_trans_x, num_trans_y, num_trans_z = adaptive_trans_num(xyz, init_dict['num_trans'], xy_only=False)
            x_points, y_points, z_points = get_starting_points(num_trans_x, num_trans_y, num_trans_z)

            trans_coords = torch.meshgrid(x_points, y_points, z_points)
            trans_arr = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()

    return trans_arr


def rot_from_ypr(ypr_array):
    # ypr_array is assumed to have a shape of [3, ]
    yaw, pitch, roll = ypr_array
    yaw = yaw.unsqueeze(0)
    pitch = pitch.unsqueeze(0)
    roll = roll.unsqueeze(0)

    tensor_0 = torch.zeros(1, device=yaw.device)
    tensor_1 = torch.ones(1, device=yaw.device)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                    torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return R

def write_summaries(writer, scalar_summaries, step):
    for (k, v) in scalar_summaries.items():
        v = np.array(v).mean().item()
        writer.add_scalar(k, v, step)
    scalar_summaries = defaultdict(list)


def trim_input_loss(img: torch.Tensor, xyz: torch.Tensor, rgb: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, num_input: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trim translation starting point & rotation by comparing sampling loss values

    Args:
        img: (H, W, 3) torch tensor containing RGB values of the image
        xyz: (N, 3) torch tensor containing xyz coordinates of the point cloud
        rgb: (N, 3) torch tensor containing RGB values of the point cloud
        trans: (K, 3) torch tensor containing translation starting point candidates
        rot: (K, 3) torch tensor containing starting rotation candidates (yaw component)
        num_input: number to trim starting translation & rotation

    Returns:
        trimmed_trans: (num_input, 3) torch tensor containing trimmed translation starting point
        trimmed_rot: (num_input) torch tensor containing trimmed rotation (yaw component)
    """

    img = img.clone().detach()
    H, W, _ = img.shape
    loss_table = torch.zeros((len(trans), len(rot)), device=img.device)

    with tqdm(desc="Loss Initialization", total=len(trans) * len(rot)) as pbar:
        for i in range(len(trans)):
            for j in range(len(rot)):
                # rotation matrix
                R = rot_from_ypr(rot[j])

                new_xyz = xyz.t() - trans[i].reshape(3, -1)
                new_xyz = (torch.matmul(R, new_xyz)).t()

                coord_arr = cloud2idx(new_xyz)
                sample_rgb = sample_from_img(img, coord_arr)
                mask = torch.sum(sample_rgb == 0, dim=1) != 3
                rgb_loss = torch.norm(sample_rgb[mask] - rgb[mask], dim=-1).mean()

                loss_table[i, j] = rgb_loss

                pbar.update(1)

    num_input = min(num_input, len(loss_table.flatten()))
    min_inds = loss_table.flatten().argsort()[:num_input]

    trimmed_trans = trans[min_inds // len(rot)]
    trimmed_rot = rot[min_inds % len(rot)]

    return trimmed_trans, trimmed_rot


def trim_input_hist_secondary(img: torch.Tensor, xyz: torch.Tensor, rgb: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor,
        num_input: int, num_split_h: int, num_split_w: int) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Trim translation starting point & rotation by comparing color histogram intersection

    Args:
        img: (H, W, 3) torch tensor containing RGB values of the image
        xyz: (N, 3) torch tensor containing xyz coordinates of the point cloud
        rgb: (N, 3) torch tensor containing RGB values of the point cloud
        trans: (K, 3) torch tensor containing translation starting point candidates
        rot: (K, 3) torch tensor containing starting rotation candidates (yaw component)
        num_input: number to trim starting translation & rotation
        num_split_h: Number of split along horizontal direction
        num_split_w: Number of split along vertical direction

    Returns:
        trimmed_trans: (num_input, 3) torch tensor containing trimmed translation starting point
        trimmed_rot: (num_input) torch tensor containing trimmed rotation (yaw component)
    """

    num_bins = [8, 8, 8]

    img = img.clone().detach() * 255
    H, W, _ = img.shape

    # masking coordinates to remove pixels whose RGB value is [0, 0, 0]
    img_mask = torch.zeros([H, W], dtype=torch.bool, device=img.device)
    img_mask[torch.sum(img == 0, dim=2) != 3] = True

    # histograms are made from split images, then split histogram intersection is summed
    hist_intersect = torch.zeros((len(trans)), device=img.device)
    hist_intersect_split = torch.zeros(num_split_h * num_split_w, device=img.device)
    block_size_h = img.shape[0] // num_split_h
    block_size_w = img.shape[1] // num_split_w

    with tqdm(desc="Hist Initialization", total=len(trans)) as pbar:
        for i in range(len(trans)):
            # rotation matrix
            R = rot_from_ypr(rot[i])

            # make panorama from xyz, rgb
            proj_img = make_pano(torch.transpose(torch.matmul(R, torch.transpose(xyz - trans[i], 0, 1)), 0, 1), rgb, resolution=(img.shape[0], img.shape[1]), return_torch=True)
            proj_mask = torch.zeros([proj_img.shape[0], proj_img.shape[1]], dtype=torch.bool, device=img.device)
            proj_mask[torch.sum(proj_img == 0, dim=2) != 3] = True

            for h in range(1, num_split_h - 1):
                for w in range(num_split_w):
                    # masking split section
                    block_mask = torch.zeros([proj_img.shape[0], proj_img.shape[1]], dtype=torch.bool, device=img.device)
                    block_mask[h * block_size_h: (h + 1) * block_size_h, w * block_size_w: (w + 1) * block_size_w] = True
                    final_mask = torch.logical_and(proj_mask, img_mask)
                    final_mask = torch.logical_and(final_mask, block_mask)
                    final_img_mask = torch.logical_and(img_mask, block_mask)

                    tgt_proj_rgb = proj_img[torch.nonzero(final_mask, as_tuple=True)]
                    gt_proj_rgb = img[torch.nonzero(final_img_mask, as_tuple=True)]

                    # Account for full masks
                    if len(tgt_proj_rgb) == 0 or len(gt_proj_rgb) == 0:
                        hist_intersect_split[h * num_split_w + w] = 0.0
                        break

                    proj_hist = histogram(proj_img, final_mask, num_bins)
                    img_hist = histogram(img, final_img_mask, num_bins)
                    hist_intersect_split[h * num_split_w + w] = histogram_intersection(img_hist, proj_hist)

            # consider NaN
            hist_intersect_split[torch.isnan(hist_intersect_split)] = 0.
            hist_intersect[i] = hist_intersect_split.sum().item() / (num_split_h * num_split_w)

            pbar.update(1)

    min_inds = hist_intersect.flatten().argsort()[-num_input:]
    min_inds = torch.flip(min_inds, [0])
    trimmed_trans = trans[min_inds]
    trimmed_rot = rot[min_inds]

    return trimmed_trans, trimmed_rot


def make_input(img: torch.Tensor, xyz: torch.Tensor, rgb: torch.Tensor, num_input: int, init_dict=None, criterion: str = 'histogram',
        num_intermediate: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make translation & rotation starting point

    Args:
        img: (H, W, 3) torch tensor containing RGB values of the image
        xyz: (N, 3) torch tensor containing xyz coordinates of the point cloud data
        rgb: (N, 3) torch tensor containing RGB values of the point cloud data
        num_start_trans: number of translation candidates
        num_start_rot: number of rotation candidates
        num_input: number of translation, rotation starting point
        criterion: criterion to use for evaluating candidates
        num_intermediate: if criterion is 'loss_hist', num_intermediate is used for trim_input_loss
        init_dict: Dictionary containing information for initialization

    Returns:
        input_trans: (num_input, 3) torch tensor containing starting translation points
        input_rot: (num_input, 1) torch tensor containing starting rotation
    """

    # rotation candidates
    rot = generate_rot_points(init_dict, device=img.device)

    # translation candidates
    trans = generate_trans_points(xyz, init_dict, device=img.device)

    if init_dict['sample_rate_for_init'] is not None:
        mask = torch.bernoulli(torch.zeros(len(xyz), dtype=torch.bool, device=img.device), p=1 / init_dict['sample_rate_for_init'])
        input_xyz = xyz.clone().detach()[mask]
    else:
        input_xyz = xyz

    # trim candidates
    if criterion == 'loss_histogram':
        trimmed_trans, trimmed_rot = trim_input_loss(img, input_xyz, rgb, trans, rot, num_intermediate)
        input_trans, input_rot = trim_input_hist_secondary(img, input_xyz, rgb, trimmed_trans, trimmed_rot, num_input, init_dict['num_split_h'], init_dict['num_split_w'])

    return input_trans, input_rot


def reshape_img_tensor(img: torch.Tensor, size: Tuple):
    # Note that size is (X, Y)
    cv_img = (img.cpu().numpy() * 255).astype(np.uint8)
    cv_img = cv2.resize(cv_img, size)
    cv_img = cv_img / 255.

    return torch.from_numpy(cv_img).float().to(img.device)


def debug_visualize(tgt_tensor):
    """
    Visualize target tensor. If batch dimension exists, visualizes the first instance. Multi-channel inputs are shown as 'slices'.
    If number of channels is 3, displayed in RGB. Otherwise results are shown as single channel images.
    For inputs that are float, we assume that the tgt_tensor values are normalized within [0, 1].
    For inputs that are int, we assume that the tgt_tensor values are normalized within [0, 255].

    Args:
        tgt_tensor: torch.tensor with one of the following shapes: (H, W), (H, W, C), (B, H, W, C)

    Returns:
        None
    """
    if "torch" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.cpu().float().numpy()
    elif "numpy" in str(type(tgt_tensor)):
        vis_tgt = tgt_tensor.astype(np.float)
    else:
        raise ValueError("Invalid input!")

    if vis_tgt.max() > 2.0:  # If tgt_tensor is in range greater than 2.0, we assume it is an RGB image
        vis_tgt /= 255.

    if len(vis_tgt.shape) == 2:
        H, W = vis_tgt.shape
        plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())
        plt.show()

    elif len(vis_tgt.shape) == 3:
        H, W, C = vis_tgt.shape

        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())

        plt.show()

    elif len(vis_tgt.shape) == 4:
        B, H, W, C = vis_tgt.shape
        vis_tgt = vis_tgt[0]

        if C > 3 or C == 2:
            fig = plt.figure(figsize=(50, 50))
            for i in range(C):
                fig.add_subplot(C // 2, 2, i + 1)
                plt.imshow(vis_tgt[..., i], cmap='gray', vmin=vis_tgt[..., i].min(), vmax=vis_tgt[..., i].max())
        elif C == 3:  # Display as RGB
            plt.imshow(vis_tgt)
        elif C == 1:
            plt.imshow(vis_tgt, cmap='gray', vmin=vis_tgt.min(), vmax=vis_tgt.max())

        plt.show()


# Code excerpted from https://github.com/haruishi43/equilib
def create_coordinate(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    return:
        coordinate: numpy.ndarray
    """
    xs = torch.linspace(0, w_out - 1, w_out, device=device)
    theta = np.pi - xs * 2 * math.pi / w_out
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    phi, theta = torch.meshgrid([phi, theta])
    coord = torch.stack((theta, phi), axis=-1)
    return coord


def compute_sampling_grid(ypr, num_split_h, num_split_w):
    """
    Utility function for computing sampling grid using yaw, pitch, roll
    We assume the equirectangular image to be splitted as follows:

    -------------------------------------
    |   0    |   1    |    2   |    3   |
    |        |        |        |        |
    -------------------------------------
    |   4    |   5    |    6   |    7   |
    |        |        |        |        |
    -------------------------------------

    Indices are assumed to be ordered in compliance to the above convention.
    Args:
        ypr: torch.tensor of shape (3, ) containing yaw, pitch, roll
        num_split_h: Number of horizontal splits
        num_split_w: Number of vertical splits

    Returns:
        grid: Sampling grid for generating rotated images according to yaw, pitch, roll
    """
    R = rot_from_ypr(ypr).T

    H, W = num_split_h, num_split_w
    a = create_coordinate(H, W, ypr.device)
    a[..., 0] -= np.pi / (num_split_w)  # Add offset to align sampling grid to each pixel center
    a[..., 1] += np.pi / (num_split_h * 2)  # Add offset to align sampling grid to each pixel center
    norm_A = 1
    x = norm_A * torch.sin(a[:, :, 1]) * torch.cos(a[:, :, 0])
    y = norm_A * torch.sin(a[:, :, 1]) * torch.sin(a[:, :, 0])
    z = norm_A * torch.cos(a[:, :, 1])
    A = torch.stack((x, y, z), dim=-1)  # (H, W, 3)
    _B = R @ A.unsqueeze(3)
    _B = _B.squeeze(3)
    grid = cloud2idx(_B.reshape(-1, 3)).reshape(H, W, 2)
    return grid
