#!/usr/bin/env python3
"""
下载 COCO 图像数据集用于多模态性能测试
默认下载 400 张图像
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def resize_images(images_dir, target_size=(1920, 1080), allow_upscale=False):
    """
    将目录中的所有图片resize到指定尺寸（保持宽高比，填充黑边）
    
    Args:
        images_dir: 图片目录
        target_size: 目标尺寸 (width, height)，默认1080P
        allow_upscale: 是否允许放大图片，默认False（仅缩小）
    """
    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print("没有找到图片文件")
        return
    
    print()
    print("=" * 80)
    print("调整图片尺寸")
    print("=" * 80)
    print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
    print(f"图片数量: {len(image_files)}")
    print(f"允许放大: {'是' if allow_upscale else '否（仅缩小）'}")
    print(f"填充方式: 黑边填充（保持内容完整）")
    print("=" * 80)
    print()
    
    resized_count = 0
    skipped_count = 0
    upscaled_count = 0
    
    for img_path in tqdm(image_files, desc="调整尺寸"):
        try:
            with Image.open(img_path) as img:
                original_size = img.size
                
                # 计算保持宽高比的新尺寸
                width_ratio = target_size[0] / original_size[0]
                height_ratio = target_size[1] / original_size[1]
                ratio = min(width_ratio, height_ratio)
                
                # 判断是否需要调整
                need_resize = False
                if ratio < 1:  # 需要缩小
                    need_resize = True
                    resized_count += 1
                elif ratio > 1 and allow_upscale:  # 需要放大且允许放大
                    need_resize = True
                    upscaled_count += 1
                elif original_size != target_size:  # 尺寸不匹配但不需要缩放
                    need_resize = True
                    skipped_count += 1
                else:
                    skipped_count += 1
                
                if need_resize or original_size != target_size:
                    # 计算缩放后的尺寸
                    if ratio < 1 or (ratio > 1 and allow_upscale):
                        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        img_resized = img.copy()
                    
                    # 创建黑色背景
                    background = Image.new('RGB', target_size, (0, 0, 0))
                    
                    # 计算居中位置
                    paste_x = (target_size[0] - img_resized.size[0]) // 2
                    paste_y = (target_size[1] - img_resized.size[1]) // 2
                    
                    # 将图片粘贴到黑色背景上
                    if img_resized.mode == 'RGBA':
                        background.paste(img_resized, (paste_x, paste_y), img_resized)
                    else:
                        background.paste(img_resized, (paste_x, paste_y))
                    
                    # 保存
                    background.save(img_path, quality=95, optimize=True)
                    
        except Exception as e:
            print(f"\n处理失败 {img_path.name}: {e}")
            continue
    
    print()
    print(f"✓ 缩小图片: {resized_count} 张")
    if allow_upscale:
        print(f"✓ 放大图片: {upscaled_count} 张")
    print(f"  未改变: {skipped_count} 张")
    print("=" * 80)


def download_coco_images(output_dir='./dataset', num_images=400):
    """
    从 COCO 数据集下载图像
    
    Args:
        output_dir: 输出目录
        num_images: 下载的图像数量（默认 400）
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COCO 数据集下载工具")
    print("=" * 80)
    print(f"输出目录: {output_path.absolute()}")
    print(f"目标数量: {num_images} 张图像")
    print("=" * 80)
    print()
    
    try:
        from datasets import load_dataset
        
        print("加载 COCO 数据集...")
        # 使用 HuggingFace 上的 COCO 数据集
        dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
        
        print(f"开始下载 {num_images} 张图像...")
        print()
        
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        metadata = []
        images_downloaded = 0
        
        for i, item in enumerate(tqdm(dataset, total=num_images, desc="下载图像")):
            if images_downloaded >= num_images:
                break
            
            try:
                image = item['image']
                image_id = item.get('image_id', i)
                
                # 保存图像
                image_filename = f"coco_{image_id:012d}.jpg"
                image_path = images_dir / image_filename
                image.save(image_path)
                
                # 创建元数据条目
                metadata.append({
                    "image": image_filename,
                    "image_id": image_id,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "请详细描述这张图片的内容。"
                        },
                        {
                            "from": "gpt",
                            "value": ""
                        }
                    ]
                })
                
                images_downloaded += 1
                
            except Exception as e:
                print(f"\n跳过图像 {i}: {e}")
                continue
        
        print()
        print(f"✓ 成功下载 {images_downloaded} 张图像到: {images_dir}")
        
        # 保存元数据 JSON
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 元数据已保存: {metadata_file}")
        
        # 显示统计信息
        print()
        print("=" * 80)
        print("下载完成统计")
        print("=" * 80)
        print(f"图像目录: {images_dir}")
        print(f"图像数量: {images_downloaded}")
        print(f"元数据文件: {metadata_file}")
        print(f"元数据条数: {len(metadata)}")
        print("=" * 80)
        
        # 询问是否调整图片尺寸到1080P
        print()
        resize_choice = input("是否将所有图片调整到1080P (1920x1080)？(y/N): ").strip().lower()
        
        if resize_choice in ['y', 'yes']:
            resize_images(images_dir, (1920, 1080), allow_upscale=True)
        else:
            print("跳过图片调整")
        
        return True
        
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("运行: pip install datasets")
        return False
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset(dataset_dir='./dataset'):
    """
    验证数据集完整性
    
    Args:
        dataset_dir: 数据集目录
    """
    
    dataset_path = Path(dataset_dir)
    
    print("=" * 80)
    print("验证数据集")
    print("=" * 80)
    
    # 检查图像目录
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        print(f"✗ 图像目录不存在: {images_dir}")
        return False
    
    # 统计图像数量
    image_files = list(images_dir.glob("*.jpg"))
    print(f"✓ 图像目录: {images_dir}")
    print(f"  图像数量: {len(image_files)}")
    
    # 检查元数据文件
    metadata_file = dataset_path / "metadata.json"
    if not metadata_file.exists():
        print(f"✗ 元数据文件不存在: {metadata_file}")
        return False
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✓ 元数据文件: {metadata_file}")
    print(f"  条目数量: {len(metadata)}")
    
    # 验证图像和元数据匹配
    metadata_images = {item['image'] for item in metadata}
    actual_images = {img.name for img in image_files}
    
    missing = metadata_images - actual_images
    extra = actual_images - metadata_images
    
    if missing:
        print(f"⚠ 元数据中有 {len(missing)} 张图像文件缺失")
    
    if extra:
        print(f"⚠ 有 {len(extra)} 张图像未在元数据中")
    
    if not missing and not extra:
        print("✓ 图像和元数据完全匹配")
    
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='下载 COCO 图像数据集')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='输出目录 (默认: ./dataset)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=400,
        help='下载的图像数量 (默认: 400)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='仅验证现有数据集，不下载'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # 仅验证
        verify_dataset(args.output_dir)
    else:
        # 下载数据集
        success = download_coco_images(
            output_dir=args.output_dir,
            num_images=args.num_images
        )
        
        if success:
            # 验证下载的数据集
            print()
            verify_dataset(args.output_dir)
