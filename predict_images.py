import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import get_model, to_img_format, to_img, ResumeArgumentParser, HEIGHT, WIDTH, IMG_FORMAT, COOR_TH, write_pred_csv, write_pred_video


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    

def predict_from_images(image_dir, tracknet_file, inpaintnet_file='', batch_size=16, eval_mode='weight', 
                        output_video=False, traj_len=8, save_dir='pred_result', img_format='png'):
    """
    预测图像序列中的羽毛球轨迹
    
    Args:
        image_dir (str): 包含图像的目录路径
        tracknet_file (str): TrackNet模型检查点文件路径
        inpaintnet_file (str): InpaintNet模型检查点文件路径
        batch_size (int): 推理批次大小
        eval_mode (str): 评估模式
        output_video (bool): 是否输出带轨迹的视频
        traj_len (int): 轨迹长度
        save_dir (str): 保存结果的目录
        img_format (str): 图像格式
    """
    
    # 检查图像目录
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    # 按名称排序
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x)
    print(f"Found {len(image_files)} images: {image_files}")
    
    # 加载模型
    tracknet_ckpt = torch.load(tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if inpaintnet_file:
        inpaintnet_ckpt = torch.load(inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    # 读取图像
    frame_list = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        frame_list.append(np.array(img))
    
    # 获取图像尺寸
    h, w = frame_list[0].shape[:2]
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # 测试TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    
    # 创建数据集
    dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                             frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().cuda()
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()
        
        # 预测
        tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

    # 如果使用InpaintNet
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        # 创建数据集
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
            coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
            with torch.no_grad():
                coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
            
            # 阈值处理
            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.
            
            # 预测
            tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                inpaint_pred_dict[key].extend(tmp_pred[key])

    # 写入CSV文件
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取图像目录名作为输出文件名
    img_dir_name = os.path.basename(os.path.normpath(image_dir))
    out_csv_file = os.path.join(save_dir, f'{img_dir_name}_ball.csv')
    
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)
    
    print(f"Prediction results saved to {out_csv_file}")
    
    # 打印预测结果
    print("Prediction results:")
    for i in range(len(pred_dict['Frame'])):
        frame = pred_dict['Frame'][i]
        x = pred_dict['X'][i]
        y = pred_dict['Y'][i]
        vis = pred_dict['Visibility'][i]
        print(f"Frame {frame}: X={x}, Y={y}, Visibility={vis}")

    # 如果需要生成视频
    if output_video:
        # 创建视频 - 使用原始图像作为基础
        video_frames = frame_list.copy()
        video_file = os.path.join(save_dir, f'{img_dir_name}_prediction.mp4')
        write_pred_video_with_images(video_frames, pred_dict, video_file, traj_len=traj_len)
        print(f"Prediction video saved to {video_file}")

    return pred_dict

def write_pred_video_with_images(frame_list, pred_dict, save_file, traj_len=8):
    """
    使用图像列表和预测结果创建带轨迹的视频
    """
    import cv2
    from collections import deque
    from utils.general import draw_traj

    # 获取图像尺寸
    h, w = frame_list[0].shape[:2]
    fps = 30  # 设置帧率为30帧/秒，可以根据需要调整

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    # 创建轨迹队列
    traj_queue = deque()

    # 遍历每一帧
    for i, frame in enumerate(frame_list):
        # 将RGB转换为BGR（OpenCV格式）
        if frame.shape[-1] == 3:  # 如果是RGB格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # 检查轨迹队列大小
        if len(traj_queue) >= traj_len:
            traj_queue.pop()

        # 找到当前帧的预测结果
        current_pred = None
        for j, frame_id in enumerate(pred_dict['Frame']):
            if frame_id == i:
                x_pred = pred_dict['X'][j]
                y_pred = pred_dict['Y'][j]
                vis_pred = pred_dict['Visibility'][j]
                if vis_pred:  # 如果球是可见的
                    current_pred = [x_pred, y_pred]
                break

        # 添加当前预测到轨迹队列
        if current_pred is not None:
            traj_queue.appendleft(current_pred)
        else:
            traj_queue.appendleft(None)

        # 在帧上绘制轨迹
        frame_with_traj = draw_traj(frame_bgr, traj_queue, color='yellow')

        # 写入视频
        out.write(frame_with_traj)

    # 释放资源
    out.release()


def predict_from_all_subfolders(root_dir, tracknet_file, inpaintnet_file='', batch_size=16, eval_mode='weight',
                                output_video=False, traj_len=8, save_dir='pred_result', img_format='png'):
    """
    预测根目录下所有子文件夹中的图像序列
    
    Args:
        root_dir (str): 包含多个子文件夹的根目录路径
        tracknet_file (str): TrackNet模型检查点文件路径
        inpaintnet_file (str): InpaintNet模型检查点文件路径
        batch_size (int): 推理批次大小
        eval_mode (str): 评估模式
        output_video (bool): 是否输出带轨迹的视频
        traj_len (int): 轨迹长度
        save_dir (str): 保存结果的目录
        img_format (str): 图像格式
    """
    print(f"Processing all subfolders in: {root_dir}")
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, f))]
    
    if not subfolders:
        print(f"No subfolders found in {root_dir}")
        return
    
    print(f"Found {len(subfolders)} subfolders: {subfolders}")
    
    # 为每个子文件夹运行预测
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        print(f"\nProcessing subfolder: {subfolder_path}")
        
        try:
            # 为每个子文件夹创建独立的保存目录
            subfolder_save_dir = os.path.join(save_dir, subfolder)
            
            predict_from_images(
                image_dir=subfolder_path,
                tracknet_file=tracknet_file,
                inpaintnet_file=inpaintnet_file,
                batch_size=batch_size,
                eval_mode=eval_mode,
                output_video=output_video,
                traj_len=traj_len,
                save_dir=subfolder_save_dir,
                img_format=img_format
            )
            
            print(f"Completed processing: {subfolder}")
        except Exception as e:
            print(f"Error processing subfolder {subfolder}: {str(e)}")
            continue
    
    print(f"\nCompleted processing all subfolders in {root_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='directory path of the input images')
    parser.add_argument('--tracknet_file', type=str, required=True, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--img_format', type=str, default='png', help='image format (png, jpg, etc.)')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    parser.add_argument('--process_subfolders', action='store_true', default=False, help='whether to process all subfolders in the image_dir')
    args = parser.parse_args()

    if args.process_subfolders:
        predict_from_all_subfolders(
            root_dir=args.image_dir,
            tracknet_file=args.tracknet_file,
            inpaintnet_file=args.inpaintnet_file,
            batch_size=args.batch_size,
            eval_mode=args.eval_mode,
            output_video=args.output_video,
            traj_len=args.traj_len,
            save_dir=args.save_dir,
            img_format=args.img_format
        )
    else:
        predict_from_images(
            image_dir=args.image_dir,
            tracknet_file=args.tracknet_file,
            inpaintnet_file=args.inpaintnet_file,
            batch_size=args.batch_size,
            eval_mode=args.eval_mode,
            output_video=args.output_video,
            traj_len=args.traj_len,
            save_dir=args.save_dir,
            img_format=args.img_format
        ) 
        
        
# 1. 处理单个图像文件夹：
# python predict_images.py --image_dir path/to/your/images --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir results --output_video

# 2. 处理图像文件夹中的所有子文件夹：
# python predict_images.py --image_dir path/to/your/images --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir results --output_video --process_subfolders

# 3. 完整参数示例：
# python predict_images.py \
#   --image_dir /path/to/images \
#   --tracknet_file /path/to/tracknet_model.pth \
#   --inpaintnet_file /path/to/inpaintnet_model.pth \
#   --batch_size 16 \
#   --eval_mode weight \
#   --save_dir results \
#   --output_video \
#   --traj_len 8