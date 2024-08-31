from __future__ import division
import math
import os
import numpy as np
from PIL import Image
import imageio
import cv2
import torch
from tools import ops
import os
import torch
import numpy as np
import cv2
from torchvision.utils import save_image

def save_results_as_image_and_video(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration, output_dir, opt):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图片 (例如，保存第一个batch的第一个帧)
    save_image(data[0], os.path.join(output_dir, f'data_{iteration}.png'))
    save_image(y_pred_before_refine[0], os.path.join(output_dir, f'y_pred_before_refine_{iteration}.png'))
    save_image(y_pred[0], os.path.join(output_dir, f'y_pred_{iteration}.png'))

    # 将流和掩码转换为适合保存的格式
    flow = flow[0].cpu().detach().numpy()
    mask_fw = mask_fw[0].cpu().detach().numpy()
    mask_bw = mask_bw[0].cpu().detach().numpy()

    def save_as_video(tensor, filename, fps=10):
        # 确保 tensor 是 [batch_size, num_frames, channels, height, width] 的形状
        batch_size, num_frames, channels, height, width = tensor.size()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = tensor[0, i].cpu().detach().numpy().transpose(1, 2, 0)  # 取第一个batch中的第i帧
            frame = (frame * 255).astype(np.uint8)
            out.write(frame)

        out.release()

    save_as_video(data, os.path.join(output_dir, f'data_{iteration}.avi'))
    save_as_video(y_pred_before_refine, os.path.join(output_dir, f'y_pred_before_refine_{iteration}.avi'))
    save_as_video(y_pred, os.path.join(output_dir, f'y_pred_{iteration}.avi'))

    def save_flow_as_video(flow, filename, fps=10):
        # 假设 flow 的形状为 [batch_size, num_frames, 2, height, width]
        # 在此假设 flow 已经通过 permute 调整为 [batch_size, height, width, num_frames, 2]
        _flow = flow.permute(0, 2, 3, 4, 1).cpu().data.numpy()  # 变换为 [batch_size, height, width, num_frames, 2]

        batch_size, height, width, num_frames, _ = _flow.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame_id in range(num_frames):
            # 提取每一帧的光流
            frame_flow = _flow[0, :, :, frame_id, :]  # 取第一个batch中的第frame_id帧
            magnitude, angle = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out.write(rgb)

        out.release()

    save_flow_as_video(flow, os.path.join(output_dir, f'flow_{iteration}.avi'))

    # 保存掩码为视频
    def save_mask_as_video(mask, filename, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_shape = mask.shape
        out = cv2.VideoWriter(filename, fourcc, fps, (video_shape[-1], video_shape[-2]))

        for i in range(mask.shape[1]):
            frame = (mask[i] * 255).astype(np.uint8)
            out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        out.release()

    save_mask_as_video(mask_fw, os.path.join(output_dir, f'mask_fw_{iteration}.avi'))
    save_mask_as_video(mask_bw, os.path.join(output_dir, f'mask_bw_{iteration}.avi'))
def dynamic_size(num_images):
    """动态计算合适的size参数"""
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)
    return [rows, cols]


def save_images(images, size, image_path):
    num_images = size[0] * size[1]
    puzzle = merge(images[0:num_images], size)
    im = Image.fromarray(np.uint8(puzzle))

    # 确保目录存在
    directory = os.path.dirname(image_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    im.save(image_path)


def save_gif(images, length, size, gifpath):
    num_images = size[0] * size[1]
    images = np.array(images[0:num_images])
    savegif = [np.uint8(merge(images[:, times, :, :, :], size)) for times in range(0, length)]

    # 确保目录存在
    directory = os.path.dirname(gifpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    imageio.mimsave(gifpath, savegif, fps=int(length))


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


def save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration, sampledir, opt, eval=False,
                 useMask=True, single=False, bidirectional=False):
    frame1 = data[:, 0, :, :, :]
    num_predicted_frames = y_pred.size()[1] - 1
    num_frames = y_pred.size()[1]
    size = dynamic_size(num_frames)

    if useMask:
        save_gif(mask_fw.unsqueeze(4).data.cpu().numpy() * 255., num_predicted_frames, size,
                 os.path.join(sampledir, '{:06d}_foward_occ_map.gif'.format(iteration)))
        save_gif(mask_bw.unsqueeze(4).data.cpu().numpy() * 255., num_predicted_frames, size,
                 os.path.join(sampledir, '{:06d}_backward_occ_map.gif'.format(iteration)))

    frame1_ = torch.unsqueeze(frame1, 1)
    if bidirectional:
        fakegif_before_refinement = torch.cat(
            [y_pred_before_refine[:, 0:3, ...], frame1_.cuda(), y_pred_before_refine[:, 3::, ...]], 1)
    else:
        fakegif_before_refinement = torch.cat([frame1_.cuda(), y_pred_before_refine], 1)
    fakegif_before_refinement = fakegif_before_refinement.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    if bidirectional:
        fakegif = torch.cat([y_pred[:, 0:3, ...], frame1_.cuda(), y_pred[:, 3::, ...]], 1)
    else:
        fakegif = torch.cat([frame1_.cuda(), y_pred], 1)
    fakegif = fakegif.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    _flow = flow.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    if eval:
        save_file_name = 'sample'
        if bidirectional:
            data = data[:, [1, 2, 3, 0, 4, 5, 6, 7], ...].cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        else:
            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        save_gif(data * 255, opt.num_frames, size, os.path.join(sampledir, '{:06d}_gt.gif'.format(iteration)))
    else:
        save_file_name = 'recon'

    save_file_name = save_file_name.replace("..", "")
    save_gif(fakegif * 255, num_frames, size,
             os.path.join(sampledir, '{:06d}_{}.gif'.format(iteration, save_file_name)))

    save_gif(fakegif_before_refinement * 255, num_frames, size,
             os.path.join(sampledir, '{:06d}_{}_bf_refine.gif'.format(iteration, save_file_name)))

    ops.save_flow_sequence(_flow, num_predicted_frames, opt.input_size, size,
                           os.path.join(sampledir, '{:06d}_{}_flow.gif'.format(iteration, save_file_name)))

    if single:
        for i in range(5):
            imageio.imwrite(os.path.join(sampledir, '{:06d}_{}.png'.format(iteration, str(i))), fakegif[0, i, ...])


def save_images(root_dir, data, y_pred, paths, opt):
    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy() * 255  # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1] + 1)]
        aux_dir = os.path.join(root_dir, paths[0][i][0:-22])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '.gif'), frames_fo_save, fps=int(len(paths) * 2))

        for j, frame in enumerate(frames_fo_save):
            cv2.imwrite(os.path.join(root_dir, paths[0][i][0:-4] + '{:02d}.png'.format(j)),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_images_ucf(root_dir, data, y_pred, paths, opt):
    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy() * 255  # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1] + 1)]
        aux_dir = os.path.join(root_dir, paths[0][i])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i] + '.gif'), frames_fo_save, fps=int(len(paths) * 2))

        for j, frame in enumerate(frames_fo_save):
            cv2.imwrite(os.path.join(root_dir, paths[0][i], '{:02d}.png'.format(j)),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_images_kitti(root_dir, data, y_pred, paths, opt):
    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy() * 255  # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1] + 1)]
        aux_dir = os.path.join(root_dir, paths[i])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[i] + '.gif'), frames_fo_save, fps=int(len(paths) * 2))

        for j, frame in enumerate(frames_fo_save):
            frame = cv2.resize(frame, (256, 78))
            cv2.imwrite(os.path.join(root_dir, paths[i], '{:02d}.png'.format(j)),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_flows(root_dir, flow, paths):
    _flow = flow.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    for i in range(flow.size()[0]):

        flow_fo_save = [np.uint8(ops.compute_flow_color_map(_flow[i][frame_id])) for frame_id in range(len(paths) - 1)]

        aux_dir = os.path.dirname(os.path.join(root_dir, paths[0][i][0:-4] + '.gif'))
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '.gif'), flow_fo_save, fps=int(len(paths) - 1 - 2))

        for j in range(flow.size()[2]):
            ops.saveflow(_flow[i][j], (256, 128), os.path.join(root_dir, paths[j + 1][i]))


def save_occ_map(root_dir, mask, paths):
    mask = mask.data.cpu().numpy() * 255.
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            aux_dir = os.path.dirname(os.path.join(root_dir, paths[j + 1][i]))
            if not os.path.isdir(aux_dir):
                os.makedirs(aux_dir)
            cv2.imwrite(os.path.join(root_dir, paths[j + 1][i]), mask[i][j])


def save_samples_no_flow(data, y_pred, iteration, sampledir, opt, eval=False, single=False, bidirectional=False):
    frame1 = data[:, 0, :, :, :]
    num_frames = y_pred.size()[1]
    size = dynamic_size(num_frames)

    frame1_ = torch.unsqueeze(frame1, 1)

    if bidirectional:
        fakegif = torch.cat([y_pred[:, 0:3, ...], frame1_.cuda(), y_pred[:, 3::, ...]], 1)
    else:
        fakegif = torch.cat([frame1_.cuda(), y_pred], 1)
    fakegif = fakegif.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    if eval:
        save_file_name = 'sample'
        if bidirectional:
            data = data[:, [1, 2, 3, 0, 4, 5, 6, 7], ...].cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        else:
            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        save_gif(data * 255, opt.num_frames, size, os.path.join(sampledir, '{:06d}_gt.gif'.format(iteration)))
    else:
        save_file_name = 'recon'

    save_gif(fakegif * 255, num_frames, size,
             os.path.join(sampledir, '{:06d}_{}.gif'.format(iteration, save_file_name)))

    if single:
        for i in range(5):
            imageio.imwrite(os.path.join(sampledir, '{:06d}_{}.png'.format(iteration, str(i))), fakegif[0, i, ...])
