from typing import Dict, Iterator, List
import base64
import math
import numpy as np
import cv2
import json
from PIL import Image
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset

# qwen2 vl hyper params
# IMAGE_FACTOR = 28
# MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 16384 * 28 * 28
# MAX_RATIO = 200
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200



@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            messages = json.loads(item)

            images = messages['images']
            # assert messages['conversations'][0]['from'] == 'system'
            assert messages['conversations'][0]['from'] == 'user'
            
            
            # sys_prompt = messages['conversations'][0]['value']
            usr_prompt = messages['conversations'][0]['value']
            
            # 验证图片数量与<image>标记数量是否匹配
            image_placeholders_count = usr_prompt.count('<image>')
            if image_placeholders_count != len(images):
                print(f"警告: 图片数量({len(images)})与<image>标记数量({image_placeholders_count})不匹配，跳过此样本")
                continue

            # 图片预处理
            imgs = []
            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 无法读取图片 {img_path}，跳过此样本")
                    break
                imgs.append(img)
            
            # 如果有图片读取失败，跳过此样本
            if len(imgs) != len(images):
                continue
            
            imgs = [dynamic_processess_qwen2(img, 32, 3136, 262144) for img in imgs]
            imgs = [b64encode_image(cv2.imencode('.png', image)[1].tobytes()) for image in imgs]

            placeholders = [{
                "type": "image_url",
                "image_url":  {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
            } for image in imgs]

            usr_prompt_parts = usr_prompt.split('<image>')

            # 交错处理数据
            content = []
            debug_print_content = []
            
            for i in range(len(usr_prompt_parts)):
                if i < len(usr_prompt_parts) - 1:   # len(parts) - 1 可以分到图片
                    content.append({"type": "text", "text": usr_prompt_parts[i]})
                    content.append(placeholders[i])
                    
                    debug_print_content.append({"type": "text", "text": usr_prompt_parts[i]})
                    debug_print_content.append(images[i])
                else:   # 最后一个，是文本
                    content.append({"type": "text", "text": usr_prompt_parts[i]})
                    debug_print_content.append({"type": "text", "text": usr_prompt_parts[i]})

            prompt = [
                # {
                #     "role": "system",
                #     "content": sys_prompt,
                # },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            
            debug_prompt = [
                # {
                #     "role": "system",
                #     "content": sys_prompt,
                # },
                {
                    "role": "user",
                    "content": debug_print_content,
                },
            ]
            # print(debug_prompt)
            
            # import pdb; pdb.set_trace()
            
            yield prompt

# Function to encode the image
def b64encode_image(buffer):
    return base64.b64encode(buffer).decode('utf-8')


def dynamic_processess_qwen2(image, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    image = Image.fromarray(image)
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image = image.resize((resized_width, resized_height))
    image = np.array(image)
    return image


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



if __name__ == '__main__':
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    parallel = [1, 2, 4, 8, 16, 32, 64, 128]  # 并发数列表
    #parallel = [1,2]
    number = [v*10 for v in parallel]  # 每个并发数对应的测试次数

    # 使用混合数据集（包含单图和多图）
    dataset_path = './test_dataset_mixed.jsonl'  # 混合：70%单图 + 30%多图
    # dataset_path = '/home/intel/llm_test/ocr_qwen_vl/test_dataset.jsonl'  # 仅单图测试
    # dataset_path = '/home/intel/llm_test/ocr_qwen_vl/test_dataset_multi.jsonl'  # 仅多图测试
    args = Arguments(
        # model='Qwen3-VL-4B-Instruct', 
        model='Qwen3-VL-8B-Instruct', # For Intel B60
        # url='http://10.48.2.128:30096/v1/chat/completions', # 5090 tp 1
        # url='http://10.48.2.128:30086/v1/chat/completions', # 5090 tp 2
        # url='http://10.48.1.135:30087/v1/chat/completions',   # l4 tp 2
        url='http://127.0.0.1:8000/v1/chat/completions',   # Intel B60
        dataset_path=dataset_path,  # 自定义数据集路径
        api_key='EMPTY',
        dataset='custom',  # 自定义数据集名称
        headers={
            'Content-Type': 'application/json',
            'Host': 'hw-problem-solving-grader-test'
        },
        number=number,  # 测试次数
        parallel=parallel,  # 并发数
        seed=42,          # 固定随机种子，确保跨平台输入一致
        top_p=1,
        # max_tokens=24,   # 固定最大输出长度，确保跨并发测试输出长度一致
        max_tokens=1024,   # 固定最大输出长度，确保跨并发测试输出长度一致
        stream=False,
        debug=True
    )

    run_perf_benchmark(args)

