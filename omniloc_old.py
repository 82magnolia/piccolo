import torch
import numpy as np
from torch import sin, cos
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import cloud2idx, refine_img, sample_from_img, quantile, make_pano, draw_grad
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import os
import cv2


def omniloc(img, xyz, rgb, input_trans, input_rot, starting_point, cfg, scalar_summaries, 
        edge_img=None, room_name=None, log_dir=None, img_weight=None, pcd_weight=None):
    # xyz, rgb are (N, 3) input point clouds
    # img is (H, W, 3) image tensor
    
    translation = input_trans[starting_point].unsqueeze(0).t().requires_grad_()
    yaw, pitch, roll = input_rot[starting_point]
    yaw = yaw.unsqueeze(0).requires_grad_()
    roll = roll.unsqueeze(0).requires_grad_()
    pitch = pitch.unsqueeze(0).requires_grad_()

    tensor_0 = torch.zeros(1, device=xyz.device)
    tensor_1 = torch.ones(1, device=xyz.device)

    # Get configs
    lr = getattr(cfg, 'lr', 0.1)
    num_iter = getattr(cfg, 'num_iter', 100)
    patience = getattr(cfg, 'patience', 5)
    factor = getattr(cfg, 'factor', 0.9)
    vis = getattr(cfg, 'visualize', False)
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)
    filter_dist_thres = getattr(cfg, 'filter_dist_thres', None)  # Filter points only within distance threshold

    if filter_dist_thres is not None:
        dist = torch.norm(xyz - translation.t(), dim=-1)
        in_xyz = xyz[dist < filter_dist_thres].detach().clone()
    else:
        in_xyz = xyz.detach().clone()
    
    final_optimizer = torch.optim.Adam([translation, yaw, roll, pitch], lr=lr)

    loss = 0.0

    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', patience=patience, factor=factor)
    
    frames = []

    if getattr(cfg, 'filter_start', False):
        with torch.no_grad():
            RX = torch.stack([
                            torch.stack([tensor_1, tensor_0, tensor_0]),
                            torch.stack([tensor_0, cos(roll), -sin(roll)]),
                            torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

            RY = torch.stack([
                            torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

            RZ = torch.stack([
                            torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                            torch.stack([sin(yaw), cos(yaw), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            new_xyz = torch.transpose(in_xyz, 0, 1) - translation
            new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

            coord_arr = cloud2idx(new_xyz)

            filter_factor = getattr(cfg, 'filter_factor', 1)
            filtered_idx = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), rgb, quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))

            in_xyz = in_xyz[filtered_idx]
            in_rgb = rgb[filtered_idx]
    else:
        in_rgb = rgb

    if getattr(cfg, 'use_jacobian', False):
        loss_func = JacSamplingLoss(in_xyz, in_rgb, img, xyz.device, cfg)
    else:
        loss_func = SamplingLoss(in_xyz, in_rgb, img, edge_img, xyz.device, cfg, img_weight, pcd_weight)

    for iteration in tqdm(range(num_iter), desc="Starting point {}".format(starting_point)):
        final_optimizer.zero_grad()
        loss = loss_func(translation, yaw, pitch, roll)
        loss.backward()

        final_optimizer.step()
        final_scheduler.step(loss)

        with torch.no_grad():
            x_min, x_max = quantile(xyz[:, 0], out_quantile)
            y_min, y_max = quantile(xyz[:, 1], out_quantile)
            z_min, z_max = quantile(xyz[:, 2], out_quantile)
            translation[0] = torch.clamp(translation[0], min=x_min, max=x_max)
            translation[1] = torch.clamp(translation[1], min=y_min, max=y_max)
            translation[2] = torch.clamp(translation[2], min=z_min, max=z_max)

        if vis:
            with torch.no_grad():
                tmp_roll = roll.clone().detach()
                tmp_pitch = pitch.clone().detach()
                tmp_yaw = yaw.clone().detach()
                tmp_trans = translation.clone().detach()
                tmp_xyz = in_xyz.clone().detach()

                RX = torch.stack([
                                torch.stack([tensor_1, tensor_0, tensor_0]),
                                torch.stack([tensor_0, cos(tmp_roll), -sin(tmp_roll)]),
                                torch.stack([tensor_0, sin(tmp_roll), cos(tmp_roll)])]).reshape(3, 3)

                RY = torch.stack([
                                torch.stack([cos(tmp_pitch), tensor_0, sin(tmp_pitch)]),
                                torch.stack([tensor_0, tensor_1, tensor_0]),
                                torch.stack([-sin(tmp_pitch), tensor_0, cos(tmp_pitch)])]).reshape(3, 3)

                RZ = torch.stack([
                                torch.stack([cos(tmp_yaw), -sin(tmp_yaw), tensor_0]),
                                torch.stack([sin(tmp_yaw), cos(tmp_yaw), tensor_0]),
                                torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

                R = torch.mm(RZ, RY)
                R = torch.mm(R, RX)

                new_xyz = torch.transpose(tmp_xyz, 0, 1) - tmp_trans
                new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

                image_factor = getattr(cfg, 'image_factor', 2)
                cur_img = Image.fromarray(make_pano(new_xyz.clone().detach(), in_rgb.clone().detach(), resolution=(img.shape[0] // image_factor, img.shape[1] // image_factor)))
                gt_img = Image.fromarray(np.uint8(img.detach().cpu().numpy() * 255), 'RGB').resize((cur_img.width, cur_img.height))
                    
                vis_list = getattr(cfg, 'visualize_list', None)
                if vis_list is None:
                    new_frame = Image.new('RGB', (cur_img.width, 2 * cur_img.height))
                    new_frame.paste(gt_img, (0, 0))
                    new_frame.paste(cur_img, (0, cur_img.height))

                else:
                    new_frame = Image.new('RGB', (cur_img.width, len(vis_list) * cur_img.height))
                    curr_idx = 0
                    
                    if 'gt_img' in vis_list:
                        new_frame.paste(gt_img, (0, curr_idx))
                        curr_idx += cur_img.height
                    
                    if 'cur_img' in vis_list:
                        new_frame.paste(cur_img, (0, curr_idx))
                        curr_idx += cur_img.height
                    
                    if 'grad_img' in vis_list:
                        assert cfg.use_jacobian
                        grad_rgb = draw_grad(getattr(cfg, 'draw_grad_mode', 'translation_cos'), loss_func)
                        grad_rgb = cv2.cvtColor(cv2.applyColorMap((grad_rgb.cpu().unsqueeze(0).numpy() * 255).astype(np.uint8),
                            cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).squeeze(0)
                        grad_rgb = (torch.from_numpy(grad_rgb).float() / 255).to(new_xyz.device)
                        grad_img = make_pano(new_xyz.clone().detach(), grad_rgb, resolution=(img.shape[0] // image_factor, img.shape[1] // image_factor))
                        grad_composition = getattr(cfg, 'grad_composition', None)
                        if grad_composition == 'gt_img':
                            alpha_mask = cv2.threshold(cv2.cvtColor(grad_img, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
                            np_gt_img = cv2.cvtColor(np.array(gt_img) // 2, cv2.COLOR_RGB2GRAY)
                            grad_img = np.expand_dims(cv2.bitwise_and(np_gt_img, np_gt_img, mask=255 - alpha_mask), -1) + grad_img
                        grad_img = Image.fromarray(grad_img)
                        new_frame.paste(grad_img, (0, curr_idx))
                        curr_idx += cur_img.height
                
                if iteration == 0:
                    for i in range(4):
                        frames.append(new_frame)
                frames.append(new_frame)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(roll), -sin(roll)]),
                    torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                    torch.stack([sin(yaw), cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    if vis:
        if getattr(cfg, 'save_direct', False):
            save_mode = getattr(cfg, 'save_mode', 'gif')
            vis_starting_point = getattr(cfg, 'vis_starting_point', None)
            if vis_starting_point is None or starting_point in vis_starting_point:
                gif_name = room_name.replace('.png', '')
                gif_name += f"_{starting_point}"
                gif_save_dir = os.path.join(log_dir, 'gifs', 'save_direct')
                if not os.path.exists(gif_save_dir):
                    os.makedirs(gif_save_dir)

                if save_mode == 'gif':
                    frames[0].save(os.path.join(gif_save_dir, '{}.gif'.format(gif_name)),
                                format='gif', append_images=frames[1:], save_all=True, optimize=False,
                                duration=150, loop=0)
                elif save_mode == 'video':
                    videodims = (frames[0].size[0], frames[0].size[1])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video = cv2.VideoWriter(os.path.join(gif_save_dir, '{}.mp4'.format(gif_name)), fourcc, 10, videodims)
                    for frame in frames:
                        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                    video.release()
                elif save_mode == 'image':
                    frames[-1].save(os.path.join(gif_save_dir, '{}.png'.format(gif_name)))
            return [translation.cpu(), R.cpu(), loss.cpu()]
        else:
            gt_img = Image.fromarray(np.uint8(img.detach().cpu().numpy() * 255), 'RGB').resize((img.shape[1] // image_factor, img.shape[0] // image_factor))
            cur_img = Image.fromarray(np.zeros_like(gt_img))

            if vis_list is None:
                last_frame = Image.new('RGB', (cur_img.width, 2 * cur_img.height))
                last_frame.paste(gt_img, (0, 0))
                last_frame.paste(cur_img, (0, cur_img.height))
            else:
                curr_idx = 0
                last_frame = Image.new('RGB', (cur_img.width, len(vis_list) * cur_img.height))
                
                if 'gt_img' in vis_list:
                    last_frame.paste(gt_img, (0, curr_idx))
                    curr_idx += cur_img.height
                
                if 'cur_img' in vis_list:
                    last_frame.paste(cur_img, (0, curr_idx))
                    curr_idx += cur_img.height
                
                if 'grad_img' in vis_list:
                    last_frame.paste(grad_img, (0, curr_idx))
                    curr_idx += cur_img.height

            for i in range(10):
                frames.append(new_frame)
            for i in range(5):
                frames.append(last_frame)

            return [translation.cpu(), R.cpu(), loss.cpu(), frames]
    else:
        return [translation.cpu(), R.cpu(), loss.cpu()]


def omniloc_batch(img, xyz, rgb, input_trans, input_rot, cfg, scalar_summaries, edge_img=None):
    # xyz, rgb are (N, 3) input point clouds
    # img is (H, W, 3) image tensor
    # edge_img is (H, W, 3) image tensor
    assert cfg.num_input > 1

    batch_size = input_trans.shape[0]
    translation = input_trans.unsqueeze(-1)  # (B, 3, 1)
    yaw = input_rot[..., 0:1]  # (B, 1)
    pitch = input_rot[..., 1:2]
    roll = input_rot[..., 2:3]

    translation_list = torch.chunk(translation, batch_size)
    yaw_list = torch.chunk(yaw, batch_size)
    pitch_list = torch.chunk(pitch, batch_size)
    roll_list = torch.chunk(roll, batch_size)

    tensor_0 = torch.zeros(1, device=xyz.device)
    tensor_1 = torch.ones(1, device=xyz.device)

    # Get configs
    lr = getattr(cfg, 'lr', 0.1)
    num_iter = getattr(cfg, 'num_iter', 100)
    patience = getattr(cfg, 'patience', 5)
    factor = getattr(cfg, 'factor', 0.9)
    out_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    in_xyz = xyz.detach().clone()
    in_rgb = rgb

    if getattr(cfg, 'filter_start', False):
        with torch.no_grad():
            xyz_list = []
            rgb_list = []
            for idx in range(batch_size):
                RX = torch.stack([
                                torch.stack([tensor_1, tensor_0, tensor_0]),
                                torch.stack([tensor_0, cos(roll[idx]), -sin(roll[idx])]),
                                torch.stack([tensor_0, sin(roll[idx]), cos(roll[idx])])]).reshape(3, 3)

                RY = torch.stack([
                                torch.stack([cos(pitch[idx]), tensor_0, sin(pitch[idx])]),
                                torch.stack([tensor_0, tensor_1, tensor_0]),
                                torch.stack([-sin(pitch[idx]), tensor_0, cos(pitch[idx])])]).reshape(3, 3)

                RZ = torch.stack([
                                torch.stack([cos(yaw[idx]), -sin(yaw[idx]), tensor_0]),
                                torch.stack([sin(yaw[idx]), cos(yaw[idx]), tensor_0]),
                                torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

                R = torch.mm(RZ, RY)
                R = torch.mm(R, RX)

                new_xyz = torch.transpose(in_xyz, 0, 1) - translation[idx]
                new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

                coord_arr = cloud2idx(new_xyz)

                filter_factor = getattr(cfg, 'filter_factor', 1)
                filtered_idx = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), rgb, quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))
                xyz_list.append(in_xyz[filtered_idx])
                rgb_list.append(rgb[filtered_idx])

            max_size = max([pt.shape[0] for pt in xyz_list])
            filtered_xyz = torch.zeros([batch_size, max_size, 3], device=xyz.device)
            filtered_rgb = torch.zeros([batch_size, max_size, 3], device=xyz.device)
            filtered_mask = torch.zeros([batch_size, max_size], device=xyz.device, dtype=torch.bool)

            for idx in range(batch_size):
                filtered_xyz[idx, :xyz_list[idx].shape[0]] = xyz_list[idx]
                filtered_rgb[idx, :rgb_list[idx].shape[0]] = rgb_list[idx]
                filtered_mask[idx, :xyz_list[idx].shape[0]] = True
            
            in_xyz = filtered_xyz
            in_rgb = filtered_rgb
        loss_func = BatchSamplingLoss(in_xyz, in_rgb, img, edge_img, xyz.device, cfg, filtered_mask)
    else:
        in_rgb = rgb
        loss_func = BatchSamplingLoss(in_xyz, in_rgb, img, edge_img, xyz.device, cfg)

    optimizer_list = [torch.optim.Adam([translation_list[idx].requires_grad_(), yaw_list[idx].requires_grad_(), 
        roll_list[idx].requires_grad_(), pitch_list[idx].requires_grad_()], lr=lr) for idx in range(batch_size)]
    scheduler_list = [ReduceLROnPlateau(optimizer_list[idx], mode='min', patience=patience, factor=factor) for idx in range(batch_size)]

    translation = torch.cat(translation_list)
    yaw = torch.cat(yaw_list)
    pitch = torch.cat(pitch_list)
    roll = torch.cat(roll_list)

    with torch.no_grad():
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

    for iteration in tqdm(range(num_iter), desc="Global Step"):
        for idx in range(batch_size):
            optimizer_list[idx].zero_grad()
        
        loss, loss_list = loss_func(translation, yaw, pitch, roll)  # scalar and tensor of shape (B, )
        loss.backward()

        for idx in range(batch_size):
            optimizer_list[idx].step()
            scheduler_list[idx].step(loss_list[idx])

        translation = torch.cat(translation_list)
        yaw = torch.cat(yaw_list)
        pitch = torch.cat(pitch_list)
        roll = torch.cat(roll_list)

        with torch.no_grad():
            for idx in range(batch_size):
                translation_list[idx][0, 0, 0] = torch.clamp(translation_list[idx][0, 0, 0], min=x_min, max=x_max)
                translation_list[idx][0, 1, 0] = torch.clamp(translation_list[idx][0, 1, 0], min=y_min, max=y_max)
                translation_list[idx][0, 2, 0] = torch.clamp(translation_list[idx][0, 2, 0], min=z_min, max=z_max)

    min_idx = loss_list.argmin().item()
    translation = translation[min_idx]  # (3, 1)
    yaw = yaw[min_idx]
    pitch = pitch[min_idx]
    roll = roll[min_idx]
    loss = loss_list[min_idx]

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(roll), -sin(roll)]),
                    torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                    torch.stack([sin(yaw), cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    return [translation.cpu(), R.cpu(), loss.cpu()]


def sampling_loss(img, xyz, rgb, input_trans, input_rot, starting_point, cfg, return_list=True):
    # xyz, rgb are (N, 3) input point clouds
    # img is (H, W, 3) image tensor
    
    translation = input_trans[starting_point].unsqueeze(0).t().requires_grad_()
    yaw, pitch, roll = input_rot[starting_point]
    yaw = yaw.unsqueeze(0).requires_grad_()
    roll = roll.unsqueeze(0).requires_grad_()
    pitch = pitch.unsqueeze(0).requires_grad_()

    tensor_0 = torch.zeros(1, device=xyz.device)
    tensor_1 = torch.ones(1, device=xyz.device)

    # Get configs
    filter_dist_thres = getattr(cfg, 'filter_dist_thres', None)  # Filter points only within distance threshold

    if filter_dist_thres is not None:
        dist = torch.norm(xyz - translation.t(), dim=-1)
        in_xyz = xyz[dist < filter_dist_thres].detach().clone()
    else:
        in_xyz = xyz.detach().clone()

    if getattr(cfg, 'filter_start', False):
        with torch.no_grad():
            RX = torch.stack([
                            torch.stack([tensor_1, tensor_0, tensor_0]),
                            torch.stack([tensor_0, cos(roll), -sin(roll)]),
                            torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

            RY = torch.stack([
                            torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

            RZ = torch.stack([
                            torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                            torch.stack([sin(yaw), cos(yaw), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            new_xyz = torch.transpose(in_xyz, 0, 1) - translation
            new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

            coord_arr = cloud2idx(new_xyz)

            filter_factor = getattr(cfg, 'filter_factor', 1)
            filtered_idx = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), rgb, quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))

            in_xyz = in_xyz[filtered_idx]
            in_rgb = rgb[filtered_idx]
    else:
        in_rgb = rgb

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, cos(roll), -sin(roll)]),
                    torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
                    torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
                    torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                    torch.stack([sin(yaw), cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)

    new_xyz = torch.transpose(in_xyz, 0, 1) - translation
    new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

    coord_arr = cloud2idx(new_xyz)

    if getattr(cfg, 'filter_idx', False):
        assert not getattr(cfg, 'filter_start', False)
        filter_factor = getattr(cfg, 'filter_factor', 1)
        filtered_idx = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), rgb, quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))

        coord_arr = coord_arr[filtered_idx]
        refined_rgb = in_rgb[filtered_idx]
    else:
        refined_rgb = in_rgb

    sample_rgb = sample_from_img(img, coord_arr)
    mask = torch.sum(sample_rgb == 0, dim=1) != 3

    rgb_loss = torch.norm(sample_rgb[mask] - refined_rgb[mask], dim=-1).mean()

    loss = rgb_loss

    if return_list:
        return [translation.cpu(), R.cpu(), loss.cpu()]
    else:
        return loss.cpu()


class SamplingLoss(nn.Module):
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, edge_img: torch.tensor, device: torch.device, cfg,
            img_weight: torch.tensor, pcd_weight: torch.tensor):
        super(SamplingLoss, self).__init__()
        self.xyz = xyz
        self.rgb = rgb
        self.img = img
        self.edge_img = edge_img
        self.cfg = cfg
        self.tensor_0 = torch.zeros(1, device=xyz.device)
        self.tensor_1 = torch.ones(1, device=xyz.device)
        self.img_weight = img_weight
        self.pcd_weight = pcd_weight

    def forward(self, translation, yaw, pitch, roll):
        RX = torch.stack([
                        torch.stack([self.tensor_1, self.tensor_0, self.tensor_0]),
                        torch.stack([self.tensor_0, cos(roll), -sin(roll)]),
                        torch.stack([self.tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([cos(pitch), self.tensor_0, sin(pitch)]),
                        torch.stack([self.tensor_0, self.tensor_1, self.tensor_0]),
                        torch.stack([-sin(pitch), self.tensor_0, cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([cos(yaw), -sin(yaw), self.tensor_0]),
                        torch.stack([sin(yaw), cos(yaw), self.tensor_0]),
                        torch.stack([self.tensor_0, self.tensor_0, self.tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        new_xyz = torch.transpose(self.xyz, 0, 1) - translation
        new_xyz = torch.transpose(torch.matmul(R, new_xyz), 0, 1)

        coord_arr = cloud2idx(new_xyz)

        if getattr(self.cfg, 'filter_edge', False):
            with torch.no_grad():
                sample_edge = sample_from_img(self.edge_img.unsqueeze(-1), coord_arr).squeeze(-1)
            coord_arr = coord_arr[sample_edge > 0.8]
            in_rgb = self.rgb[sample_edge > 0.8]
            new_xyz = new_xyz[sample_edge > 0.8]
        else:
            in_rgb = self.rgb

        if getattr(self.cfg, 'filter_idx', False):
            assert not getattr(self.cfg, 'filter_start', False)
            filter_factor = getattr(self.cfg, 'filter_factor', 1)
            filtered_mask = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), in_rgb, quantization=(self.img.shape[0] // filter_factor, 
                self.img.shape[1] // filter_factor), return_valid_mask=True)
        
        sample_rgb = sample_from_img(self.img, coord_arr)
        raw_loss = torch.norm(sample_rgb - in_rgb, dim=-1)
        
        # Add weight to loss
        loss_weight_list = []
        if self.img_weight is not None:
            with torch.no_grad():
                loss_img_weight = sample_from_img(self.img_weight, coord_arr)
            loss_weight_list.append(loss_img_weight.squeeze())
        if self.pcd_weight is not None:
            loss_weight_list.append(self.pcd_weight.squeeze())

        if len(loss_weight_list) == 1:
            raw_loss = loss_weight_list[0] * raw_loss
        elif len(loss_weight_list) == 2:
            raw_loss = 0.5 * (loss_weight_list[0] + loss_weight_list[1]) * raw_loss

        if getattr(self.cfg, 'filter_idx', False):
            mask = (torch.sum(sample_rgb == 0, dim=1) != 3) & filtered_mask
            rgb_loss = (raw_loss * mask).sum() / mask.sum().float()
        else:
            mask = torch.sum(sample_rgb == 0, dim=1) != 3
            rgb_loss = (raw_loss * mask).sum() / mask.sum().float()

        return rgb_loss


class BatchSamplingLoss(nn.Module):
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, edge_img: torch.tensor, device: torch.device, cfg, filter_mask: torch.tensor = None):
        super(BatchSamplingLoss, self).__init__()
        if not getattr(cfg, 'filter_start', False):  # If filter_start is True, points are already in shape (N, N_pt, 3)
            self.xyz = xyz.expand(cfg.num_input, -1, -1)  # (N, N_pt, 3)
            self.rgb = rgb.expand(cfg.num_input, -1, -1)  # (N, N_pt, 3)
        else:
            self.xyz = xyz
            self.rgb = rgb

        self.img = img
        self.edge_img = edge_img
        self.total_img = torch.cat([self.img, self.edge_img.unsqueeze(-1)], dim=-1)
        self.cfg = cfg
        self.num_input = cfg.num_input
        self.tensor_0 = torch.zeros(self.num_input, 1, device=xyz.device)
        self.tensor_1 = torch.ones(self.num_input, 1, device=xyz.device)
        self.filter_mask = filter_mask

    def forward(self, translation, yaw, pitch, roll):
        # translation has shape (N, 3, 1)
        # yaw, pitch, roll has shape (N, 1)

        RX = torch.cat([
                        torch.stack([self.tensor_1, self.tensor_0, self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, cos(roll), -sin(roll)], dim=-1),
                        torch.stack([self.tensor_0, sin(roll), cos(roll)], dim=-1)], dim=1)
        RY = torch.cat([
                        torch.stack([cos(pitch), self.tensor_0, sin(pitch)], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_1, self.tensor_0], dim=-1),
                        torch.stack([-sin(pitch), self.tensor_0, cos(pitch)], dim=-1)], dim=1)
        RZ = torch.cat([
                        torch.stack([cos(yaw), -sin(yaw), self.tensor_0], dim=-1),
                        torch.stack([sin(yaw), cos(yaw), self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_0, self.tensor_1], dim=-1)], dim=1)

        # RX, RY, RZ: (N, 3, 3)
        R = torch.bmm(RZ, RY)
        R = torch.bmm(R, RX)

        new_xyz = self.xyz - torch.transpose(translation, 1, 2)
        
        # Faster way to bmm
        tmp_xyz = torch.zeros_like(new_xyz, device=new_xyz.device)
        tmp_xyz[..., 0] = (new_xyz * R[:, 0:1, :]).sum(-1)
        tmp_xyz[..., 1] = (new_xyz * R[:, 1:2, :]).sum(-1)
        tmp_xyz[..., 2] = (new_xyz * R[:, 2:3, :]).sum(-1)

        new_xyz = tmp_xyz

        coord_arr = cloud2idx(new_xyz, batched=True)  # (B, N, 2)

        if getattr(self.cfg, 'filter_idx', False):
            filter_factor = getattr(self.cfg, 'filter_factor', 1)
            filter_idx, valid_mask = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), self.rgb, 
                quantization=(self.img.shape[0] // filter_factor, self.img.shape[1] // filter_factor), batched=True)

            refined_rgb = self.rgb
            sample_rgb = sample_from_img(self.img, coord_arr, batched=True)  # (B, N, 3)
            rgb_mask = torch.sum(sample_rgb == 0, dim=-1) != 3  # (B, N)
            mask = valid_mask & rgb_mask  # (B, N)

        else:
            refined_rgb = self.rgb  # (B, N, 3)

            if getattr(self.cfg, 'filter_edge', False):
                sample_tot = sample_from_img(self.total_img, coord_arr, batched=True)  # (B, N, 4)
                sample_rgb = sample_tot[..., :3]  # (B, N, 3)
                sample_edge = sample_tot[..., 3]  # (B, N)

                mask = (torch.sum(sample_rgb == 0, dim=-1) != 3) & (sample_edge > 0.8)  # (B, N)
            else:
                sample_rgb = sample_from_img(self.img, coord_arr, batched=True)  # (B, N, 3)
                mask = torch.sum(sample_rgb == 0, dim=-1) != 3  # (B, N)

        if self.filter_mask is not None:
            mask &= self.filter_mask

        rgb_loss_list = torch.norm(sample_rgb - refined_rgb, dim=-1) * mask  # (B, N, )
        rgb_loss_list = rgb_loss_list.sum(-1)  # (B, )

        mask_count = mask.sum(-1)  # (B, )
        rgb_loss_list /= mask_count

        rgb_loss = rgb_loss_list.sum()
        return rgb_loss, rgb_loss_list


class JacSamplingLoss(nn.Module):
    # Sampling loss that uses jacobian-style computations
    def __init__(self, xyz: torch.tensor, rgb: torch.tensor, img: torch.tensor, device: torch.device, cfg):
        super(JacSamplingLoss, self).__init__()
        self.xyz = xyz
        self.rgb = rgb
        self.img = img
        self.cfg = cfg
        self.tensor_0 = torch.zeros(self.xyz.shape[0], 1, device=xyz.device)
        self.tensor_1 = torch.ones(self.xyz.shape[0], 1, device=xyz.device)
        self.expand_list = {"translation": None, "yaw": None, "pitch": None, "roll": None}

    def forward(self, translation, yaw, pitch, roll):
        # translation has shape (N_pt, 3)
        # yaw, pitch, roll has shape (N_pt, 1)
        expand_trans = translation.squeeze().expand(self.xyz.shape[0], -1)

        expand_yaw = yaw.expand(self.xyz.shape[0], -1)
        expand_pitch = pitch.expand(self.xyz.shape[0], -1)
        expand_roll = roll.expand(self.xyz.shape[0], -1)

        expand_trans.retain_grad()
        expand_yaw.retain_grad()
        expand_pitch.retain_grad()
        expand_roll.retain_grad()

        if getattr(self.cfg, 'retain_grad', False):
            self.expand_list['translation'] = expand_trans
            self.expand_list['yaw'] = expand_yaw
            self.expand_list['pitch'] = expand_pitch
            self.expand_list['roll'] = expand_roll

        RX = torch.cat([
                        torch.stack([self.tensor_1, self.tensor_0, self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, cos(expand_roll), -sin(expand_roll)], dim=-1),
                        torch.stack([self.tensor_0, sin(expand_roll), cos(expand_roll)], dim=-1)], dim=1)
        RY = torch.cat([
                        torch.stack([cos(expand_pitch), self.tensor_0, sin(expand_pitch)], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_1, self.tensor_0], dim=-1),
                        torch.stack([-sin(expand_pitch), self.tensor_0, cos(expand_pitch)], dim=-1)], dim=1)
        RZ = torch.cat([
                        torch.stack([cos(expand_yaw), -sin(expand_yaw), self.tensor_0], dim=-1),
                        torch.stack([sin(expand_yaw), cos(expand_yaw), self.tensor_0], dim=-1),
                        torch.stack([self.tensor_0, self.tensor_0, self.tensor_1], dim=-1)], dim=1)

        # RX, RY, RZ: (N, 3, 3)
        R = torch.bmm(RZ, RY)
        R = torch.bmm(R, RX)

        new_xyz = self.xyz - expand_trans
        new_xyz = torch.bmm(new_xyz.unsqueeze(1), torch.transpose(R, 1, 2)).squeeze()
        # new_xyz = tmp_xyz  # (N, 3)

        coord_arr = cloud2idx(new_xyz)  # (N, 2)

        if getattr(self.cfg, 'filter_idx', False):
            filter_factor = getattr(self.cfg, 'filter_factor', 1)
            valid_mask = refine_img(coord_arr, torch.norm(new_xyz, dim=-1), self.rgb, 
                quantization=(self.img.shape[0] // filter_factor, self.img.shape[1] // filter_factor), return_valid_mask=True)

            refined_rgb = self.rgb
            sample_rgb = sample_from_img(self.img, coord_arr)  # (N, 3)
            rgb_mask = torch.sum(sample_rgb == 0, dim=-1) != 3  # (N, )
            mask = valid_mask & rgb_mask  # (N, )

        else:
            refined_rgb = self.rgb  # (N, 3)

            sample_rgb = sample_from_img(self.img, coord_arr)  # (N, 3)
            mask = torch.sum(sample_rgb == 0, dim=-1) != 3  # (N, )

        rgb_loss = torch.norm(sample_rgb - refined_rgb, dim=-1) * mask  # (N, )
        rgb_loss = rgb_loss.sum(-1)  # (1, )

        mask_count = mask.sum(-1)  # (1, )
        rgb_loss /= mask_count

        return rgb_loss
