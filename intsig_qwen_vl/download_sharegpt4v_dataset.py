#!/usr/bin/env python3
"""
下载 ShareGPT4V 数据集中的 COCO 图像
"""

import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm


def download_sharegpt4v_coco(output_dir='./sharegpt4v_data', num_images=100):
    """
    下载 ShareGPT4V 数据集中的 COCO 图像
    
    Args:
        output_dir: 输出目录
        num_images: 下载的图像数量限制（None表示全部下载）
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("下载 ShareGPT4V 数据集")
    print("=" * 80)
    print(f"输出目录: {output_path.absolute()}")
    print()
    
    try:
        # 下载数据集元数据
        print("下载数据集元数据...")
        repo_id = "Lin-Chen/ShareGPT4V"
        
        # 下载 JSON 文件
        json_files = [
            "sharegpt4v_instruct_gpt4-vision_cap100k.json",
            "share-captioner_coco_lcs_sam_1246k_1107.json"
        ]
        
        json_data = []
        for json_file in json_files:
            try:
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=json_file,
                    repo_type="dataset"
                    # 不使用 local_dir，避免创建复杂的目录结构
                )
                print(f"✓ 下载: {json_file}")
                
                # 读取JSON数据
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        json_data.extend(data)
                    print(f"  包含 {len(data)} 条数据")
            except Exception as e:
                print(f"✗ 无法下载 {json_file}: {e}")
        
        print(f"\n总共加载 {len(json_data)} 条数据")
        
        # 筛选包含COCO图像的数据
        coco_data = []
        for item in json_data:
            if 'image' in item and 'coco' in item['image'].lower():
                coco_data.append(item)
        
        print(f"找到 {len(coco_data)} 条COCO图像数据")
        
        # 限制下载数量
        if num_images and len(coco_data) > num_images:
            coco_data = coco_data[:num_images]
            print(f"限制下载数量: {num_images}")
        
        # 保存筛选后的元数据
        metadata_file = output_path / "coco_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存元数据: {metadata_file}")
        
        # 不再下载图像文件（改用 COCO URL 方式）
        print("\n提示: 图像文件请使用 COCO URL 方式下载")
        print(f"已保存 {len(coco_data)} 条元数据，可用于后续图像下载")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常以触发备用下载方案


def download_coco_images_via_url(output_dir='./sharegpt4v_data', metadata_file=None, num_images=100):
    """
    通过COCO官方URL下载图像（最可靠的方案）
    """
    
    print("=" * 80)
    print("使用COCO官方URL下载图像")
    print("=" * 80)
    
    output_path = Path(output_dir)
    images_dir = output_path / "coco" / "train2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取元数据获取需要的图像ID
    image_ids_to_download = []
    
    if metadata_file and Path(metadata_file).exists():
        print(f"从元数据文件读取图像列表: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for item in metadata[:num_images]:
            if 'image' in item:
                # 提取图像ID，例如从 "coco/train2017/000000000009.jpg" 提取 "000000000009"
                image_path = item['image']
                if 'train2017' in image_path:
                    image_name = os.path.basename(image_path)
                    image_id = image_name.replace('.jpg', '')
                    image_ids_to_download.append(image_id)
        
        print(f"需要下载 {len(image_ids_to_download)} 张图像")
    else:
        # 如果没有元数据，生成一些随机的COCO图像ID
        print("生成随机COCO图像ID...")
        import random
        for _ in range(num_images):
            image_id = f"{random.randint(1, 600000):012d}"
            image_ids_to_download.append(image_id)
    
    # COCO图像下载URL模板
    coco_url_template = "http://images.cocodataset.org/train2017/{}.jpg"
    
    print("\n开始下载图像...")
    downloaded = 0
    failed = 0
    failed_urls = []
    
    try:
        import requests
        from tqdm import tqdm
        
        for image_id in tqdm(image_ids_to_download, desc="下载COCO图像"):
            image_filename = f"{image_id}.jpg"
            image_path = images_dir / image_filename
            
            # 如果文件已存在且大小>0，跳过
            if image_path.exists() and image_path.stat().st_size > 0:
                downloaded += 1
                continue
            
            try:
                # 下载图像
                url = coco_url_template.format(image_id)
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200 and len(response.content) > 0:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                else:
                    failed += 1
                    failed_urls.append((image_id, response.status_code))
                    
            except Exception as e:
                failed += 1
                failed_urls.append((image_id, str(e)))
                continue
        
        print(f"\n下载完成:")
        print(f"  成功: {downloaded}")
        print(f"  失败: {failed}")
        print(f"  图像目录: {images_dir.absolute()}")
        
        if failed > 0 and failed_urls:
            print(f"\n前5个失败的图像ID:")
            for img_id, error in failed_urls[:5]:
                print(f"  {img_id}: {error}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def download_coco_sample_images(output_dir='./sharegpt4v_data', num_images=100):
    """
    直接从COCO数据集下载示例图像（备用方案）
    """
    
    print("=" * 80)
    print("使用备用方案：从COCO数据集下载示例图像")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        print("加载COCO数据集...")
        # 使用HuggingFace上的COCO数据集
        dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
        
        print(f"下载前 {num_images} 张图像...")
        
        images_downloaded = 0
        for i, item in enumerate(dataset):
            if images_downloaded >= num_images:
                break
            
            try:
                image = item['image']
                image_id = item.get('image_id', i)
                
                # 保存图像
                image_path = output_path / f"coco_{image_id:012d}.jpg"
                image.save(image_path)
                
                images_downloaded += 1
                if images_downloaded % 10 == 0:
                    print(f"已下载: {images_downloaded}/{num_images}")
                    
            except Exception as e:
                print(f"跳过图像 {i}: {e}")
                continue
        
        print(f"\n完成！下载了 {images_downloaded} 张图像到: {output_path.absolute()}")
        
        # 创建简单的元数据
        metadata = []
        for img_file in sorted(output_path.glob("*.jpg"))[:num_images]:
            metadata.append({
                "image": img_file.name,
                "conversations": [
                    {"from": "human", "value": "请描述这张图片的内容。"},
                    {"from": "gpt", "value": ""}
                ]
            })
        
        metadata_file = output_path.parent / "coco_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ 创建元数据文件: {metadata_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='下载ShareGPT4V或COCO图像数据集')
    parser.add_argument('--output-dir', type=str, default='./sharegpt4v_data',
                        help='输出目录')
    parser.add_argument('--num-images', type=int, default=100,
                        help='下载的图像数量')
    parser.add_argument('--use-sharegpt4v', action='store_true',
                        help='使用ShareGPT4V下载（需要特殊权限）')
    parser.add_argument('--metadata-file', type=str,
                        help='现有的元数据文件路径')
    
    args = parser.parse_args()
    
    if args.use_sharegpt4v:
        # 尝试从ShareGPT4V下载
        try:
            download_sharegpt4v_coco(
                output_dir=args.output_dir,
                num_images=args.num_images
            )
        except Exception as e:
            print(f"\nShareGPT4V下载失败: {e}")
            print("尝试使用COCO官方URL下载...")
            metadata_file = f"{args.output_dir}/coco_metadata.json"
            download_coco_images_via_url(
                output_dir=args.output_dir,
                metadata_file=metadata_file,
                num_images=args.num_images
            )
    else:
        # 默认：先下载元数据，再使用COCO官方URL下载图片
        print("步骤1: 下载ShareGPT4V元数据...")
        try:
            download_sharegpt4v_coco(
                output_dir=args.output_dir,
                num_images=args.num_images
            )
        except:
            pass  # 元数据下载失败不要紧，可能已经存在
        
        # 步骤2: 使用COCO官方URL下载图片
        print("\n步骤2: 使用COCO官方URL下载图片...")
        metadata_file = args.metadata_file or f"{args.output_dir}/coco_metadata.json"
        download_coco_images_via_url(
            output_dir=args.output_dir,
            metadata_file=metadata_file,
            num_images=args.num_images
        )
