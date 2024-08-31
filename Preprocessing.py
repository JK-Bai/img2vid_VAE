import os
import cv2
import random
from shutil import move, rmtree
from sklearn.model_selection import train_test_split

# 输入和输出文件夹路径
input_folders = [
    "D:/ImageToVedio/ucf/PlayingCello",
    "D:/ImageToVedio/ucf/PlayingDaf",
    "D:/ImageToVedio/ucf/PlayingFlute",
    "D:/ImageToVedio/ucf/PlayingGuitar",
    "D:/ImageToVedio/ucf/PlayingPiano",
    "D:/ImageToVedio/ucf/PlayingSitar",
    "D:/ImageToVedio/ucf/PlayingTabla",
    "D:/ImageToVedio/ucf/PlayingViolin",
    # 继续添加你选择的文件夹...
]
output_folder = "D:/ImageToVedio/output"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")

# 创建输出文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


# 函数：保存视频
def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


# 遍历每个文件夹
for folder in input_folders:
    videos = []
    for filename in os.listdir(folder):
        if filename.endswith((".mp4", ".avi")):  # 支持更多格式
            video_path = os.path.join(folder, filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"无法读取视频文件: {video_path}")
                continue

            frames = []
            success, frame = cap.read()
            while success:
                frames.append(frame)
                if len(frames) == 30:
                    videos.append(frames)
                    frames = []
                success, frame = cap.read()

            cap.release()

    # 划分为训练集和测试集
    train_videos, test_videos = train_test_split(videos, test_size=0.2, random_state=42)

    # 保存训练集视频
    for i, frames in enumerate(train_videos):
        video_name = f"{os.path.basename(folder)}_train_{i}.mp4"
        save_video(frames, os.path.join(train_folder, video_name))

    # 保存测试集视频
    for i, frames in enumerate(test_videos):
        video_name = f"{os.path.basename(folder)}_test_{i}.mp4"
        save_video(frames, os.path.join(test_folder, video_name))


# 遍历所有视频，检查是否能够读取
def check_videos(folder):
    for filename in os.listdir(folder):
        video_path = os.path.join(folder, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"删除无法读取的视频文件: {video_path}")
            os.remove(video_path)
        cap.release()


check_videos(train_folder)
check_videos(test_folder)

print("视频处理完成。")
