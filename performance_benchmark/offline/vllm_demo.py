import vllm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import glob
import time
import common_args
# from torch.profiler import profile, ProfilerActivity
# activaties = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

def build_messages(prompt_template, images, metadata=dict, system_prompt=()):
    content = list()
    system = dict(role="system", content=system_prompt)
    for image in images:
        content.append(dict(type="image", image=image))
    content.append(dict(type="text", text=prompt_template))
    user = dict(role="user", content=content)
    messages = [system, user]
    return messages

def warmup(model, sampling_params, text, images):
    vllm_inputs = [
        {
            "prompt": text,
            "multi_modal_data": {
                "image": [images[0]]
            }
        }
    ]
    results = model.generate(prompts=vllm_inputs, sampling_params=sampling_params)
    output_text = [result.outputs[0].text for result in results]
    del results
    torch.cuda.empty_cache()

def batch_inference(model, sampling_params, text, images, batch):
    num_iter = 1
    # # num_iter = 1
    # if num_iter == 0:
    #     num_iter = 1
    vllm_inputs = [
        {
            "prompt": text,
            "multi_modal_data": {
                "image": [image]
            }
        } for image in images[:batch]
    ]
    start = time.perf_counter()
    for _ in range(num_iter):
        results = model.generate(prompts=vllm_inputs, sampling_params=sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    throughput = num_iter * batch / (end - start)
    # throughput = f'{batch} throughput: ', num_iter * batch / (end - start)
    # # print(f'{batch} throughput: ', num_iter * batch / (end - start))
    output_text = [result.outputs[0].text for result in results]
    for line in output_text:
        print(line)
    print(len(output_text))
    del vllm_inputs
    torch.cuda.empty_cache()
    # return throughput
    return throughput

def run_qwen2_5_vl(args):
    # default: Load the model on the available device(s)
    results = []
    systemPrompt = []
    if args.system_prompt:
        systemPrompt = ["You are a helpful assistant."]
    model_path = args.model_path
    imgs_path = args.imgs_path
    
    images = [Image.open(path) for path in glob.glob(imgs_path + "/*")]
    # for i in range(len(images)):
    #     images[i] = images[i].resize((384, 224))
    model = LLM(
        model=model_path,
        max_model_len=8192 if not 'llava' in model_path else 4096,
        max_num_seqs=128,
        tensor_parallel_size=1,
        # max_parallel_loading_workers=16,
        # max_num_batched_tokens=16384*16,
        # gpu_memory_utilization=0.5,
        # pipeline_parallel_size=2,
        # quantization="AWQ",
        quantization=None,
        limit_mm_per_prompt={"image": 16},
        enforce_eager=True if args.enforce_eager else False,
        # mm_processor_kwargs={
        #     "min_pixels": 256*28*28,
        #     "max_pixels": 1280*28*28,
        # },
        # enable_chunked_prefill=True
    )

    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.05,
        temperature=0,
        top_p=0.01,
        top_k=-1,
        min_p=0.0,
        seed=None,
        max_tokens=256,
        stop_token_ids=[],
    )
    # default processer

    processor = AutoProcessor.from_pretrained(model_path)
    image_content = [{"type": "image", "image": image} for image in images]
    video_content = {"type": "video", "video": images}

    text = "Please describe the image"
    messages = build_messages(text, images[:1], {}, system_prompt=systemPrompt)
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    warmup(model, sampling_params, text, images)
    results.append(f"image shape is {str(images[0].size)}")
    throughput = batch_inference(model, sampling_params, text, images, 1)
    print(f"batch 1 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 2)
    print(f"batch 2 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 4)
    print(f"batch 4 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 8)
    print(f"batch 8 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 16)
    print(f"batch 16 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 32)
    print(f"batch 32 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 64)
    print(f"batch 64 throughput is {throughput} it/s")
    throughput = batch_inference(model, sampling_params, text, images, 128)
    print(f"batch 128 throughput is {throughput} it/s")
    del model, vllm_inputs

if __name__ == "__main__":
    args = common_args.parse_args()
    run_qwen2_5_vl(args)

    
