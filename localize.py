import torch
import numpy as np
import random
import cv2
import os
import time
import csv
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from typing import NamedTuple
from collections import defaultdict
from color_utils import color_mod, color_match
from utils import *
import data_utils
from omniloc import omniloc, sampling_loss, omniloc_batch


def get_init_dict(cfg: NamedTuple):
    xy_only = getattr(cfg, 'xy_only', True)
    num_trans = getattr(cfg, 'num_trans', 50)
    yaw_only = getattr(cfg, 'yaw_only', True)
    num_yaw = getattr(cfg, 'num_yaw', 4)
    num_pitch = getattr(cfg, 'num_pitch', 0)
    num_roll = getattr(cfg, 'num_roll', 0)

    max_yaw = getattr(cfg, 'max_yaw', 2 * np.pi)
    min_yaw = getattr(cfg, 'min_yaw', 0)
    max_pitch = getattr(cfg, 'max_pitch', 2 * np.pi)
    min_pitch = getattr(cfg, 'min_pitch', 0)
    max_roll = getattr(cfg, 'max_roll', 2 * np.pi)
    min_roll = getattr(cfg, 'min_roll', 0)

    x_max = getattr(cfg, 'x_max', None)
    x_min = getattr(cfg, 'x_min', None)
    y_max = getattr(cfg, 'y_max', None)
    y_min = getattr(cfg, 'y_min', None)
    z_max = getattr(cfg, 'z_max', None)
    z_min = getattr(cfg, 'z_min', None)

    z_prior = getattr(cfg, 'z_prior', None)
    dataset = cfg.dataset
    sample_rate_for_init = getattr(cfg, 'sample_rate_for_init', None)
    trans_init_mode = getattr(cfg, 'trans_init_mode', 'quantile')

    num_split_h = getattr(cfg, 'num_split_h', 2)
    num_split_w = getattr(cfg, 'num_split_w', 4)

    init_dict = {'xy_only': xy_only,
        'num_trans': num_trans,
        'yaw_only': yaw_only,
        'num_yaw': num_yaw,
        'num_pitch': num_pitch,
        'num_roll': num_roll,
        'max_yaw': max_yaw,
        'min_yaw': min_yaw,
        'max_pitch': max_pitch,
        'min_pitch': min_pitch,
        'max_roll': max_roll,
        'min_roll': min_roll,
        'z_prior': z_prior,
        'dataset': dataset,
        'sample_rate_for_init': sample_rate_for_init,
        'trans_init_mode': trans_init_mode,
        'x_max': x_max,
        'x_min': x_min,
        'y_max': y_max,
        'y_min': y_min,
        'z_max': z_max,
        'z_min': z_min,
        'num_split_h': num_split_h,
        'num_split_w': num_split_w}

    return init_dict


def localize_stanford(cfg: NamedTuple, writer: torch.utils.tensorboard.SummaryWriter, log_dir: str):
    """
    Main function for performing localization in Stanford2D-3D-S dataset.

    Args:
        cfg: Config file
        writer: SummaryWriter for writing logs
        log_dir: Directory in which logs will be saved
    
    Returns:
        None
    """
    vis = getattr(cfg, 'visualize', False)
    area_num = getattr(cfg, 'area', None)
    out_of_room_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)
    eval_full = getattr(cfg, 'eval_full', False)

    scalar_summaries = defaultdict(list)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if area_num is not None:
        if type(area_num) == list:
            filenames = []
            for each_area_num in area_num:
                filenames += sorted(glob("./data/stanford/pano/area_{}/*.png".format(each_area_num)),
                                    key=lambda x: (x.split('/')[-1].split('_')[2], int(x.split('/')[-1].split('_')[3])))
        else:
            filenames = sorted(glob("./data/stanford/pano/area_{}/*.png".format(area_num)),
                               key=lambda x: (x.split('/')[-1].split('_')[2], int(x.split('/')[-1].split('_')[3])))
    else:
        filenames = sorted(glob("./data/stanford/pano/area_*/*.png"),
                           key=lambda x: (int(x.split('/')[-2].replace('area_', '')), x.split('/')[-1].split('_')[2], int(x.split('/')[-1].split('_')[3])))
    
    room_name = getattr(cfg, 'room_name', None)
    sample_rate = getattr(cfg, 'sample_rate', 1)

    if room_name is not None:
        filenames = [file_name for file_name in filenames if room_name in file_name]

    well_posed = 0
    total_img = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    failed_lists = []
    wrong_gt_lists = []
    room_list = []

    past_pcd_name = ""
    summary = open(os.path.join(log_dir, 'stanford_results.csv'), 'w', encoding='utf-8', newline='')
    summary_writer = csv.writer(summary)
    summary_writer.writerow(['area_num', 'pano_name', 'gt_trans', 'gt_rot', 'skipped?', 'OmniLoc_trans', 'OmniLoc_rot', 't_error (m)', 'r_error (degrees)', 'time (s)'])

    # Optionally resize image
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    # Check if point cloud is aligned in gravity direction
    gravity_aligned = getattr(cfg, 'gravity_aligned', True)

    for trial, filename in enumerate(filenames):

        area_num = int(filename.split('/')[-2].split('_')[-1])
        img_name = filename.split('/')[-1]
        room_type = img_name.split('_')[2]
        room_no = img_name.split('_')[3]

        # Delete inefficient loading
        pcd_name = "./data/stanford/pcd_not_aligned/area_{}/{}_{}.txt".format(area_num, room_type, room_no)
        if past_pcd_name != pcd_name:
            xyz_np, rgb_np = data_utils.read_stanford(pcd_name, sample_rate)

            if not gravity_aligned:
                align_trans, align_rot = data_utils.obtain_align_matrix(xyz_np)
                xyz_np = np.transpose(np.matmul(align_rot, np.transpose(xyz_np) - align_trans))
            
            xyz = torch.from_numpy(xyz_np).float()
            rgb = torch.from_numpy(rgb_np).float()
            past_pcd_name = pcd_name

            xyz = xyz.to(device)
            rgb = rgb.to(device)
            non_sharpened_rgb = rgb.clone().detach()

        orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
        img = torch.from_numpy(img).float() / 255.
        img = img.to(device)
        non_sharpened_img = torch.from_numpy(orig_img).float().to(device) / 255.

        sharpen_color = getattr(cfg, 'sharpen_color', False)
        num_bins = getattr(cfg, 'num_bins', 256)
        if sharpen_color:
            if past_pcd_name == pcd_name:
                rgb = torch.from_numpy(rgb_np).float()
                rgb = rgb.to(device)
            img, rgb = color_mod(img, rgb, num_bins)

        gt_trans, gt_rot = data_utils.obtain_gt_stanford(area_num, img_name)
        
        if not gravity_aligned:
            # TODO: Make pose predictions align to the original point cloud instead (for inference)S
            gt_trans = np.matmul(align_rot, gt_trans - align_trans)
            gt_rot = np.matmul(gt_rot, align_rot.T)

        gt_trans = torch.from_numpy(gt_trans).float()
        gt_rot = torch.from_numpy(gt_rot).float()

        if out_of_room(xyz.cpu(), gt_trans, out_of_room_quantile) and not eval_full:
            print('corrupted file : {}, gt_trans is out of the room\n'.format(filename))
            wrong_gt_lists.append(filename)
            writer.add_text('skipped rooms', filename)
            summary_writer.writerow([area_num, img_name,
                                     str(gt_trans.numpy().flatten())[1:-1].replace('\n', ''),
                                     str(gt_rot.numpy().flatten())[1:-1].replace('\n', ''), 1])
            continue

        room_list.append(room_type)
        num_input = getattr(cfg, 'num_input', 6)
        num_intermediate = getattr(cfg, 'num_intermediate', 20)
        criterion = getattr(cfg, 'criterion', 'histogram')
        loss_type = getattr(cfg, 'loss_type', None)
        parallel = getattr(cfg, 'parallel', False)

        init_dict = get_init_dict(cfg)
        start_time = time.time()
        input_trans, input_rot = make_input(img, xyz, rgb, num_input, init_dict, criterion, num_intermediate)

        img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
        img = torch.from_numpy(img).float() / 255.
        img = img.to(device)

        result = []
        if parallel:
            result.append(omniloc_batch(img, xyz, rgb, input_trans, input_rot, cfg, scalar_summaries))
        else:
            for i in range(num_input):
                result.append(omniloc(img, xyz, rgb, input_trans, input_rot, i, cfg, scalar_summaries))

        end_time = time.time()
        time_spent = end_time - start_time

        with torch.no_grad():

            result = np.asarray(result, dtype=object)
            gt_trans = gt_trans.numpy()
            gt_rot = gt_rot.numpy()
            
            min_ind = result[:, 2].argmin()
            t = (result[:, 0])[min_ind]
            r = (result[:, 1])[min_ind]

            print('\n' + img_name)
            print("min_index : {}".format(min_ind))
            print("min loss : {}".format(result[min_ind, 2]))

            t_error = np.linalg.norm(gt_trans - np.array(t.clone().detach().cpu()))
            print("translation error : {}".format(t_error))
            
            r_error = np.trace(np.matmul(np.transpose(r.clone().detach().cpu()), gt_rot))
            if r_error < -1:
                r_error = -2 - r_error
            elif r_error > 3:
                r_error = 6 - r_error
            r_error = np.rad2deg(np.abs(np.arccos((r_error - 1) / 2)))
            print("rotation error : {}\n".format(r_error))

            if (t_error < 0.2) and (r_error < np.rad2deg(0.2)):
                well_posed += 1
            else:
                failed_lists.append(filename)
                writer.add_text('failed rooms', filename)
            total_img += 1
            accuracy = well_posed / total_img
            scalar_summaries['current_accuracy'] += [accuracy]
            print("current accuracy : {} ({}/{})\n".format(accuracy, well_posed, total_img))

            summary_writer.writerow([area_num, img_name, str(gt_trans.flatten())[1:-1].replace('\n', ''),
                                     str(gt_rot.flatten())[1:-1].replace('\n', ''), 0,
                                     str(t.cpu().numpy().flatten())[1:-1].replace('\n', ''),
                                     str(r.cpu().numpy().flatten())[1:-1].replace('\n', ''),
                                     t_error, r_error, time_spent])

            new_xyz = torch.transpose(xyz.cpu(), 0, 1) - t
            new_xyz = torch.transpose(torch.matmul(r, new_xyz), 0, 1)

            new_img = make_pano(new_xyz.clone().detach(), non_sharpened_rgb.cpu(), resolution=(img.shape[0] // 2, img.shape[1] // 2))
            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            img_save_dir = os.path.join(log_dir, 'results', 'area_{}'.format(area_num))
            gif_save_dir = os.path.join(log_dir, 'gifs', 'area_{}'.format(area_num))
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            
            gt_img = cv2.resize((non_sharpened_img.clone().detach().cpu().numpy() * 255).astype(np.uint8), (new_img.shape[1], new_img.shape[0]))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            best_img = cv2.vconcat([gt_img, new_img])
            cv2.imwrite(os.path.join(img_save_dir, img_name), best_img)

            if vis:
                if not os.path.exists(gif_save_dir):
                    os.makedirs(gif_save_dir)
                gif_name = img_name.split('.')[0]
                frames = (result[:, 3])[min_ind]
                frames[0].save(os.path.join(gif_save_dir, '{}.gif'.format(gif_name)),
                               format='gif', append_images=frames[1:], save_all=True, optimize=False,
                               duration=150, loop=0)
            write_summaries(writer, scalar_summaries, trial)

    summary.close()

    writer.add_scalar('final accuracy', accuracy)
    
    print(f"Final Accuracy : {accuracy}")
    print("failed {} rooms : {}\n".format(len(failed_lists), failed_lists))
    print("skipped {} rooms : {}".format(len(wrong_gt_lists), wrong_gt_lists))


def localize_omniscenes(cfg: NamedTuple, writer: torch.utils.tensorboard.SummaryWriter, log_dir: str):
    """
    Main function for performing localization in OmniScenes dataset.
    Args:
        cfg: Config file
        writer: SummaryWriter for writing logs
        log_dir: Directory in which logs will be saved
    
    Returns:
        None
    """
    out_of_room_quantile = getattr(cfg, 'out_of_room_quantile', 0.05)

    room_name = getattr(cfg, 'room_name', None)
    scene_number = getattr(cfg, 'scene_number', None)
    split_name = getattr(cfg, 'split_name', 'extreme')
    sample_rate = getattr(cfg, 'sample_rate', 1)
    parallel = getattr(cfg, 'parallel', False)

    scalar_summaries = defaultdict(list)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    filenames = sorted(glob(f"./data/omniscenes/{split_name}_pano/*/*"))

    if room_name is not None:
        if isinstance(room_name, str):
            filenames = [file_name for file_name in filenames if room_name in file_name]
        elif isinstance(room_name, list):
            filenames = [file_name for file_name in filenames if any([rm in file_name for rm in room_name])]
    if scene_number is not None:
        filenames = [file_name for file_name in filenames if "scene_{}".format(scene_number) in file_name]

    well_posed = 0
    total_img = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    failed_lists = []
    wrong_gt_lists = []
    room_list = []

    past_pcd_name = ""
    summary = open(os.path.join(log_dir, 'omniscenes_results.csv'), 'w', encoding='utf-8', newline='')
    summary_writer = csv.writer(summary)
    summary_writer.writerow(['pano_name', 'gt_trans', 'gt_rot', 'skipped?', 'OmniLoc_trans', 'OmniLoc_rot', 't_error (m)', 'r_error (degrees)', 'time (s)'])

    # Optionally resize image
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1) // 2  # Match resolution with stanford
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1) // 2  # Match resolution with stanford
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    # Check if point cloud is aligned in gravity direction
    gravity_aligned = getattr(cfg, 'gravity_aligned', True)

    for trial, filename in enumerate(filenames):

        video_name = filename.split('/')[-2]
        img_seq = filename.split('/')[-1]
        img_name = '{}/{}'.format(video_name, img_seq)
        room_type = video_name.split('_')[1]
        room_no = video_name.split('_')[2]
        scene_no = video_name.split('_')[-1]

        # Delete inefficient loading
        pcd_name = "./data/omniscenes/pcd/{}_{}.txt".format(room_type, room_no)
        if past_pcd_name != pcd_name:
            xyz_np, rgb_np = data_utils.read_omniscenes(pcd_name, sample_rate)
            if not gravity_aligned:
                align_trans, align_rot = data_utils.obtain_align_matrix(xyz_np)
                xyz_np = np.transpose(np.matmul(align_rot, np.transpose(xyz_np) - align_trans))
            
            xyz = torch.from_numpy(xyz_np).float()
            rgb = torch.from_numpy(rgb_np).float()

            xyz = xyz.to(device)
            rgb = rgb.to(device)

        orig_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (2048, 1024))
        
        # Synthetic illumination change
        if getattr(cfg, 'synth_const', None) is not None:
            orig_img = orig_img // cfg.synth_const
        if getattr(cfg, 'synth_gamma', None) is not None:
            orig_img = (((orig_img / 255.) ** cfg.synth_gamma) * 255).astype(np.uint8)
        if getattr(cfg, 'synth_wb', None):
            orig_img[..., 0] = (((orig_img[..., 0] / 255.) * cfg.synth_r) * 255).astype(np.uint8)
            orig_img[..., 1] = (((orig_img[..., 1] / 255.) * cfg.synth_g) * 255).astype(np.uint8)
            orig_img[..., 2] = (((orig_img[..., 2] / 255.) * cfg.synth_b) * 255).astype(np.uint8)

            orig_img[orig_img > 255] = 255

        sharpen_color = getattr(cfg, 'sharpen_color', False)
        match_color = getattr(cfg, 'match_color', False)
        num_bins = getattr(cfg, 'num_bins', 256)
        
        # Color modulation
        mod_img = torch.from_numpy(orig_img).float() / 255.
        mod_img = mod_img.to(device)
        if match_color:
            new_img = color_match(mod_img, rgb)
            orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)
        if sharpen_color:
            if past_pcd_name == pcd_name:
                rgb = torch.from_numpy(rgb_np).float()
                rgb = rgb.to(device)
            new_img, rgb = color_mod(mod_img, rgb, num_bins)
            orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

        img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
        img = torch.from_numpy(img).float() / 255.
        img = img.to(device)
        color_process_img = torch.from_numpy(orig_img).float().to(device) / 255.

        gt_trans, gt_rot = data_utils.obtain_gt_omniscenes(filename)
        gt_trans = torch.from_numpy(gt_trans).float()
        gt_rot = torch.from_numpy(gt_rot).float()
 
        if out_of_room(xyz.cpu(), gt_trans, out_of_room_quantile):
            print('corrupted file : {}, gt_trans is out of the room\n'.format(filename))
            wrong_gt_lists.append(filename)
            writer.add_text('skipped rooms', filename)
            summary_writer.writerow([img_name,
                                     str(gt_trans.numpy().flatten())[1:-1].replace('\n', ''),
                                     str(gt_rot.numpy().flatten())[1:-1].replace('\n', ''), 1])
            continue

        room_list.append(room_type)
        num_input = getattr(cfg, 'num_input', 6)
        num_intermediate = getattr(cfg, 'num_intermediate', 20)
        criterion = getattr(cfg, 'criterion', 'histogram')

        init_dict = get_init_dict(cfg)
        start_time = time.time()

        # Attributes for inlier detection
        inlier_init_dict = dict(init_dict)
        inlier_init_dict['is_inlier_dict'] = True
        inlier_init_dict['num_trans'] = getattr(cfg, 'inlier_num_trans', init_dict['num_trans'])
        inlier_init_dict['num_yaw'] = getattr(cfg, 'inlier_num_yaw', 4)
        inlier_init_dict['num_pitch'] = getattr(cfg, 'inlier_num_pitch', 4)
        inlier_init_dict['num_roll'] = getattr(cfg, 'inlier_num_roll', 4)
        inlier_init_dict['trans_init_mode'] = getattr(cfg, 'inlier_trans_init_mode', 'quantile')

        # Point cloud inlier filtering for initialization
        init_input_xyz = xyz
        init_input_rgb = rgb

        # Change pcd_name, scene_no to new name
        past_pcd_name = pcd_name

        input_trans, input_rot = make_input(img, init_input_xyz, init_input_rgb, num_input, init_dict, criterion, num_intermediate)

        # Visualize input points
        if getattr(cfg, 'save_starting_point', False):
            for idx in range(num_input):
                starting_xyz = torch.transpose(xyz.cpu(), 0, 1) - input_trans[idx].unsqueeze(-1).cpu()
                starting_xyz = torch.transpose(torch.matmul(rot_from_ypr(input_rot[idx]).cpu(), starting_xyz), 0, 1)

                starting_img = make_pano(starting_xyz.clone().detach(), rgb.cpu(), resolution=(orig_img.shape[0] // 2, orig_img.shape[1] // 2))
                starting_img = cv2.cvtColor(starting_img, cv2.COLOR_RGB2BGR)
                img_save_dir = os.path.join(log_dir, 'starting_points', video_name)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                
                gt_img = cv2.resize((color_process_img.clone().detach().cpu().numpy() * 255).astype(np.uint8), (starting_img.shape[1], starting_img.shape[0]))
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
                best_img = cv2.vconcat([gt_img, starting_img])
                cv2.imwrite(os.path.join(img_save_dir, f"{img_seq.split('.')[0]}_{idx}.png"), best_img)

        img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
        img = torch.from_numpy(img).float() / 255.
        img = img.to(device)

        result = []

        if parallel:
            result.append(omniloc_batch(img, xyz, rgb, input_trans, input_rot, cfg, scalar_summaries))
        else:
            for i in range(num_input):
                result.append(omniloc(img, xyz, rgb, input_trans, input_rot, i, cfg, scalar_summaries))
 
        end_time = time.time()
        time_spent = end_time - start_time

        with torch.no_grad():
            
            result = np.asarray(result, dtype=object)
            gt_trans = gt_trans.numpy()
            gt_rot = gt_rot.numpy()
            
            min_ind = result[:, 2].argmin()
            t = (result[:, 0])[min_ind]
            r = (result[:, 1])[min_ind]

            print('\n' + filename)
            print("min_index : {}".format(min_ind))
            print("min loss : {}".format(result[min_ind, 2]))

            t_error = np.linalg.norm(gt_trans - np.array(t.clone().detach().cpu()))
            print("translation error : {}".format(t_error))
            
            r_error = np.trace(np.matmul(np.transpose(r.clone().detach().cpu()), gt_rot))
            if r_error < -1:
                r_error = -2 - r_error
            elif r_error > 3:
                r_error = 6 - r_error
            r_error = np.rad2deg(np.abs(np.arccos((r_error - 1) / 2)))
            print("rotation error : {}\n".format(r_error))

            if (t_error < 0.1) and (r_error < 5):
                well_posed += 1
            else:
                failed_lists.append(filename)
                writer.add_text('failed rooms', filename)
            total_img += 1
            accuracy = well_posed / total_img
            scalar_summaries['current_accuracy'] += [accuracy]
            print("current accuracy : {} ({}/{})\n".format(accuracy, well_posed, total_img))

            summary_writer.writerow([img_name, str(gt_trans.flatten())[1:-1].replace('\n', ''),
                                     str(gt_rot.flatten())[1:-1].replace('\n', ''), 0,
                                     str(t.cpu().numpy().flatten())[1:-1].replace('\n', ''),
                                     str(r.cpu().numpy().flatten())[1:-1].replace('\n', ''),
                                     t_error, r_error, time_spent])
            write_summaries(writer, scalar_summaries, trial)

    summary.close()

    writer.add_scalar('final accuracy', accuracy)
    
    print(f"Final Accuracy : {accuracy}")
    print("failed {} rooms\n".format(len(failed_lists)))
    print("skipped {} rooms".format(len(wrong_gt_lists)))
