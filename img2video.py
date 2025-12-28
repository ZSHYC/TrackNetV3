import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def combine_images_to_video(images_dir, output_dir, fps=30, video_format='mp4'):
    """
    将指定目录下所有子文件夹中的图片合并成视频
    
    Args:
        images_dir (str): 包含子文件夹的根目录路径
        output_dir (str): 输出视频的保存目录
        fps (int): 视频帧率
        video_format (str): 输出视频格式
    """
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, f))]
    
    if not subfolders:
        print(f"No subfolders found in {images_dir}")
        return
    
    print(f"Found {len(subfolders)} subfolders: {subfolders}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个子文件夹生成视频
    for subfolder in subfolders:
        subfolder_path = os.path.join(images_dir, subfolder)
        print(f"\nProcessing subfolder: {subfolder_path}")
        
        # 获取子文件夹中的所有图片
        image_files = [f for f in os.listdir(subfolder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {subfolder_path}")
            continue
        
        # 按名称排序
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x)
        print(f"Found {len(image_files)} images")
        
        if not image_files:
            print(f"No valid images found in {subfolder_path}")
            continue
        
        # 读取第一张图片以获取尺寸
        first_img_path = os.path.join(subfolder_path, image_files[0])
        first_img = cv2.imread(first_img_path)
        
        if first_img is None:
            print(f"Could not read first image: {first_img_path}")
            continue
        
        height, width, layers = first_img.shape
        
        # 定义视频编码器和创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
        output_video_path = os.path.join(output_dir, f"{subfolder}.{video_format}")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 将图片逐个写入视频
        for img_file in tqdm(image_files, desc=f"Creating video for {subfolder}"):
            img_path = os.path.join(subfolder_path, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Warning: Could not read image {img_path}, skipping...")
                continue
            
            # 确保所有帧的尺寸一致
            if frame.shape != (height, width, layers):
                frame = cv2.resize(frame, (width, height))
            
            video_writer.write(frame)
        
        # 释放VideoWriter
        video_writer.release()
        print(f"Video saved to: {output_video_path}")
    
    print(f"\nCompleted processing all subfolders. Videos saved to: {output_dir}")


def combine_images_to_video_advanced(images_dir, output_dir, fps=30, video_format='mp4', 
                                   target_width=None, target_height=None):
    """
    将指定目录下所有子文件夹中的图片合并成视频（高级版，支持尺寸调整）
    
    Args:
        images_dir (str): 包含子文件夹的根目录路径
        output_dir (str): 输出视频的保存目录
        fps (int): 视频帧率
        video_format (str): 输出视频格式
        target_width (int): 目标宽度（可选）
        target_height (int): 目标高度（可选）
    """
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, f))]
    
    if not subfolders:
        print(f"No subfolders found in {images_dir}")
        return
    
    print(f"Found {len(subfolders)} subfolders: {subfolders}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个子文件夹生成视频
    for subfolder in subfolders:
        subfolder_path = os.path.join(images_dir, subfolder)
        print(f"\nProcessing subfolder: {subfolder_path}")
        
        # 获取子文件夹中的所有图片
        image_files = [f for f in os.listdir(subfolder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {subfolder_path}")
            continue
        
        # 按名称排序
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x)
        print(f"Found {len(image_files)} images")
        
        if not image_files:
            print(f"No valid images found in {subfolder_path}")
            continue
        
        # 读取第一张图片以获取尺寸
        first_img_path = os.path.join(subfolder_path, image_files[0])
        first_img = cv2.imread(first_img_path)
        
        if first_img is None:
            print(f"Could not read first image: {first_img_path}")
            continue
        
        original_height, original_width = first_img.shape[:2]
        
        # 确定最终尺寸
        final_width = target_width if target_width else original_width
        final_height = target_height if target_height else original_height
        
        # 定义视频编码器和创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
        output_video_path = os.path.join(output_dir, f"{subfolder}.{video_format}")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (final_width, final_height))
        
        # 将图片逐个写入视频
        for img_file in tqdm(image_files, desc=f"Creating video for {subfolder}"):
            img_path = os.path.join(subfolder_path, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Warning: Could not read image {img_path}, skipping...")
                continue
            
            # 调整图片尺寸到目标尺寸
            if target_width and target_height:
                frame = cv2.resize(frame, (final_width, final_height))
            elif frame.shape[:2] != (final_height, final_width):
                frame = cv2.resize(frame, (final_width, final_height))
            
            video_writer.write(frame)
        
        # 释放VideoWriter
        video_writer.release()
        print(f"Video saved to: {output_video_path}")
    
    print(f"\nCompleted processing all subfolders. Videos saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine images in subfolders into videos")
    parser.add_argument('--images_dir', type=str, required=True, 
                        help='Directory containing subfolders with images')
    parser.add_argument('--output_dir', type=str, default='combined_videos', 
                        help='Directory to save the output videos')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output videos')
    parser.add_argument('--video_format', type=str, default='mp4', help='Output video format')
    parser.add_argument('--target_width', type=int, default=None, help='Target width for videos (optional)')
    parser.add_argument('--target_height', type=int, default=None, help='Target height for videos (optional)')
    
    args = parser.parse_args()
    
    if args.target_width and args.target_height:
        combine_images_to_video_advanced(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            video_format=args.video_format,
            target_width=args.target_width,
            target_height=args.target_height
        )
    else:
        combine_images_to_video(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            video_format=args.video_format
        )
        
        
        
# # 基础用法
# python combine_images_to_video.py --images_dir path/to/your/images --output_dir path/to/output

# # 指定FPS和视频格式
# python combine_images_to_video.py --images_dir path/to/your/images --output_dir path/to/output --fps 25 --video_format mp4

# # 指定输出视频尺寸
# python combine_images_to_video.py --images_dir path/to/your/images --output_dir path/to/output --target_width 1920 --target_height 1080