import math
import os
import numpy as np
import cv2
import torch
from tools import ops


def dynamic_size(num_images):
    """动态计算合适的size参数"""
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)
    return [rows, cols]


def merge(images, size):
    cdim = images.shape[-1]
    h, w = images.shape[1], images.shape[2]
    if cdim == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = np.squeeze(image)
        return img
    else:
        img = np.zeros((h * size[0], w * size[1], cdim))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img


def save_as_video(frames, video_path, fps=10):
    """保存一系列帧为视频"""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def save_videos(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration, sampledir, opt, eval=False,
                useMask=True, single=False, bidirectional=False):
    frame1 = data[:, 0, :, :, :]
    num_predicted_frames = y_pred.size()[1] - 1
    num_frames = y_pred.size()[1]
    size = dynamic_size(num_frames)

    if useMask:
        mask_fw_video = mask_fw.unsqueeze(4).data.cpu().numpy() * 255.
        mask_bw_video = mask_bw.unsqueeze(4).data.cpu().numpy() * 255.
        save_as_video([merge(mask_fw_video[:, times, :, :, :], size) for times in range(num_predicted_frames)],
                      os.path.join(sampledir, '{:06d}_foward_occ_map.avi'.format(iteration)),
                      fps=int(num_predicted_frames))
        save_as_video([merge(mask_bw_video[:, times, :, :, :], size) for times in range(num_predicted_frames)],
                      os.path.join(sampledir, '{:06d}_backward_occ_map.avi'.format(iteration)),
                      fps=int(num_predicted_frames))

    frame1_ = torch.unsqueeze(frame1, 1)
    if bidirectional:
        fakevideo_before_refinement = torch.cat(
            [y_pred_before_refine[:, 0:3, ...], frame1_.cuda(), y_pred_before_refine[:, 3::, ...]], 1)
    else:
        fakevideo_before_refinement = torch.cat([frame1_.cuda(), y_pred_before_refine], 1)
    fakevideo_before_refinement = fakevideo_before_refinement.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    if bidirectional:
        fakevideo = torch.cat([y_pred[:, 0:3, ...], frame1_.cuda(), y_pred[:, 3::, ...]], 1)
    else:
        fakevideo = torch.cat([frame1_.cuda(), y_pred], 1)
    fakevideo = fakevideo.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    _flow = flow.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    if eval:
        save_file_name = 'sample'
        if bidirectional:
            data = data[:, [1, 2, 3, 0, 4, 5, 6, 7], ...].cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        else:
            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        save_as_video([merge(data[:, times, :, :, :], size) for times in range(opt.num_frames)],
                      os.path.join(sampledir, '{:06d}_gt.avi'.format(iteration)), fps=int(opt.num_frames))
    else:
        save_file_name = 'recon'

    save_file_name = save_file_name.replace("..", "")
    save_as_video([merge(fakevideo[:, times, :, :, :], size) for times in range(num_frames)],
                  os.path.join(sampledir, '{:06d}_{}.avi'.format(iteration, save_file_name)), fps=int(num_frames))

    save_as_video([merge(fakevideo_before_refinement[:, times, :, :, :], size) for times in range(num_frames)],
                  os.path.join(sampledir, '{:06d}_{}_bf_refine.avi'.format(iteration, save_file_name)),
                  fps=int(num_frames))

    ops.save_flow_sequence(_flow, num_predicted_frames, opt.input_size, size,
                           os.path.join(sampledir, '{:06d}_{}_flow.avi'.format(iteration, save_file_name)))

    if single:
        for i in range(5):
            cv2.imwrite(os.path.join(sampledir, '{:06d}_{}.png'.format(iteration, str(i))), fakevideo[0, i, ...])

