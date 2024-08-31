import torch
import cv2
import os
from models.Vae import VAE
from tools import utils_tools
from config.opts import parse_opts
import numpy as np

# 项目路径
project_path = "D:/PycharmProjects/img2vid_VAE"

# 模型路径
model_path = os.path.join(project_path, "models/checkpoints/playingviolin_model.pth")

# 输入文件路径
input_path = os.path.join(project_path, "output/test/PlayingCello_test_0.mp4")

# 解析配置
opt = parse_opts()


def process_video(video_path, num_frames=30, input_size=(128, 128)):
    """加载视频并预处理为符合模型输入要求的格式。"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, input_size)
        frame = frame.astype(np.float32) / 255.0  # 归一化到 [0, 1]
        frames.append(frame)

    cap.release()

    # 如果帧数不足，则复制最后一帧补足到 30 帧
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    # 将数据格式转换为 [T, C, H, W]
    data = np.stack(frames, axis=0)
    data = torch.from_numpy(data).permute(0, 3, 1, 2).float()  # [T, C, H, W]

    # 第一帧和剩余帧拆分，符合模型的输入要求
    frame1 = data[0]  # 第一帧, [C, H, W]
    frame2 = data.unsqueeze(0)  # 全部帧，增加 batch 维度 [1, T, C, H, W]

    # 扩展 frame1 的 batch 维度，确保和 frame2 保持一致
    frame1 = frame1.unsqueeze(0).repeat(frame2.size(0), 1, 1, 1)  # [B, C, H, W]

    return frame1.cuda(), frame2.cuda()


def save_video(frames, output_path):
    """保存生成的视频。"""
    h, w = 128, 128
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))

    for frame in frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"视频已保存到 {output_path}")


def load_model(model_path, opt):
    """加载模型并处理可能的结构变化。"""
    vae = VAE(hallucination=False, opt=opt).cuda()
    print("加载预训练模型...")
    checkpoint = torch.load(model_path,
                            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # 模型加载方式，考虑到网络结构可能变化
    model_dict = vae.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['vae'].items() if
                       k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) < len(model_dict):
        print("部分参数未加载，可能是由于网络结构的变化。")
    model_dict.update(pretrained_dict)
    vae.load_state_dict(model_dict)
    vae.eval()
    print("模型加载完成。")
    return vae


def infer(opt, model_path, input_path):
    """使用训练好的模型进行推理，生成视频。"""
    # 加载模型
    vae = load_model(model_path, opt)

    # 处理输入视频
    print("加载输入视频并处理...")
    frame1, frame2 = process_video(input_path, num_frames=30, input_size=(128, 128))
    print(f"frame1 形状: {frame1.shape}, frame2 形状: {frame2.shape}")

    # 生成与 frame1 形状一致的噪声
    noise_bg = torch.randn(frame1.size()).cuda()

    with torch.no_grad():
        print("开始推理...")
        # 使用 frame1, frame2 和噪声作为输入
        y_pred_before_refine, y_pred, _, _, _, _, mask_fw, mask_bw = vae(frame1, frame2, noise_bg)
        generated_frames = y_pred.squeeze().cpu().numpy().transpose(0, 2, 3, 1)

    output_path = os.path.join(project_path, 'output_video.mp4')
    save_video(generated_frames, output_path)


def main():
    # 执行推理
    infer(opt, model_path, input_path)


if __name__ == '__main__':
    main()

