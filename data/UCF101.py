import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import random
import cv2
import re
import time
import numpy as np
import pdb
from scipy.misc import imread

def cv2_tensor(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.shape[0], pic.shape[1], 3)
    img = img.transpose(0, 2).transpose(1, 2).contiguous()
    return img.float().div(255)

def replace_index_and_read(image_dir, indx, size):
    new_dir = image_dir[0:-22] + str(int(image_dir[-22:-16]) + indx).zfill(6) + image_dir[-16::]
    try:
        img = cv2.resize(cv2.cvtColor(cv2.imread(new_dir, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (size[1], size[0]))
    except:
        print ('orgin_dir: ' + image_dir)
        print ('new_dir: ' + new_dir)
    # center crop
    # if img.shape[0] != img.shape[1]:
    #     frame = cv2_tensor(img[:, 64:64+128])
    # else:
    #     frame = cv2_tensor(img)
    frame = cv2_tensor(img)
    return frame

def imagetoframe(image_dir, size, num_frame):
    samples = [replace_index_and_read(image_dir, indx, size) for indx in range(num_frame)]
    return torch.stack(samples)

def complete_full_list(image_dir, num_frames, output_name):
    dir_list = [image_dir[0:-22] + str(int(image_dir[-22:-16]) + i).zfill(6) + '_' + output_name for i in range(num_frames)]
    return dir_list


def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask = np.zeros_like(prev_frame)
    hsv_mask[..., 1] = 255

    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    return mask


def load_optical_flow_mask(image_dir, size, num_frames):
    frames = [replace_index_and_read(image_dir, indx, size) for indx in range(num_frames)]
    masks = []

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1].numpy().transpose(1, 2, 0)  # 转换为HWC格式
        next_frame = frames[i].numpy().transpose(1, 2, 0)
        mask = calculate_optical_flow(prev_frame, next_frame)
        masks.append(cv2_tensor(mask))

    # 第一个帧没有光流，所以使用一个全零掩码
    masks.insert(0, torch.zeros_like(masks[0]))
    return torch.stack(masks)


class UCF101Dataset(Dataset):
    def __init__(self, datapath, datalist, num_frames=5, size=(128, 128), returnpath=False):
        self.datapath = datapath
        self.datalist = open(datalist).readlines()
        self.size = size
        self.num_frame = num_frames
        self.returnPath = returnpath

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image_dir = os.path.join(self.datapath, self.datalist[idx].strip())
        sample = imagetoframe(image_dir, self.size, self.num_frame)
        flow_mask = load_optical_flow_mask(image_dir, self.size, self.num_frame)

        if self.returnPath:
            return sample, flow_mask, complete_full_list(self.datalist[idx].strip(), self.num_frame, 'pred.png')
        else:
            return sample, flow_mask


if __name__ == '__main__':

    start_time = time.time()

    UCF101_Dataset = UCF101(datapath=CITYSCAPES_VAL_DATA_PATH, mask_data_path=CITYSCAPES_VAL_DATA_SEGMASK_PATH,
                             datalist=CITYSCAPES_VAL_DATA_MASK_LIST,
                             size=(256, 128), split='train', split_num=1,
                             num_frames=5, mask_suffix='ssmask.png', returnpath=True)

    dataloader = DataLoader(UCF101_Dataset, batch_size=32, shuffle=False, num_workers=4)

    sample, masks = iter(dataloader).next()
    print (sample.shape)
    print(masks.shape)
    pdb.set_trace()