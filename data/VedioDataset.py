import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import random


class VideoDataset(Dataset):
    def __init__(self, video_dir, num_frame=5, input_size=(128, 128), returnpath=False, normalize=True):
        self.video_dir = video_dir
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
        self.numframe = num_frame
        self.input_size = input_size
        self.returnpath = returnpath
        self.normalize = False
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(self.numframe):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.input_size)  # 调整图像尺寸
            frame = frame / 255.0  # 归一化到 [0, 1]

            if self.normalize:
                frame = (frame - self.mean) / self.std  # 基于均值和标准差的归一化

            frames.append(frame)

        cap.release()

        if len(frames) < self.numframe:
            frames += [frames[-1]] * (self.numframe - len(frames))  # 复制最后一帧以补足帧数

        data = np.stack(frames, axis=0)
        data = torch.from_numpy(data).permute(0, 3, 1, 2).float()  # 将数据格式转换为 [T, C, H, W]

        if self.returnpath:
            return data, video_path
        return data