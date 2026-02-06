import vllm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info
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
    # torch.cuda.empty_cache()
    torch.xpu.empty_cache()

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
    # torch.cuda.synchronize()
    torch.xpu.synchronize()
    end = time.perf_counter()
    throughput = num_iter * batch / (end - start)
    # throughput = f'{batch} throughput: ', num_iter * batch / (end - start)
    # # print(f'{batch} throughput: ', num_iter * batch / (end - start))
    output_text = [result.outputs[0].text for result in results]
    #for line in output_text:
    #    print(line)
    print(len(output_text))
    del vllm_inputs
    # torch.cuda.empty_cache()
    torch.xpu.empty_cache()
    # return throughput
    return throughput

def run_qwen2_5_vl(args):
    # default: Load the model on the available device(s)
    results = []
    systemPrompt = []
    # Add prompt for system...
    if args.system_prompt:
        systemPrompt = ["You are a helpful assistant."]
    model_path = args.model_path
    imgs_path = args.imgs_path
    print(f"Image path is :{imgs_path}")
    
    images = [Image.open(path) for path in glob.glob(imgs_path + "/*")]
    total_images = len(images)
    print(f"Total images loaded: {total_images}")
    # for i in range(len(images)):
    #     images[i] = images[i].resize((384, 224))
    
    # Use quantization parameter
    quant = None if args.quantization == "none" else args.quantization
    
    model = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        #device="xpu",
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=128,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        quantization=quant
        # limit_mm_per_prompt={"image": 16},
        # enforce_eager=True if args.enforce_eager else False,
        # mm_processor_kwargs={
        #     "min_pixels": 256*28*28,
        #     "max_pixels": 1280*28*28,
        # },
        # enable_chunked_prefill=True
    )
    print(f"Initialized LLM successfully")

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
    
    #warmup(model, sampling_params, text, images)
    results.append(f"image shape is {str(images[0].size)}")
    batch_size = args.batch
    
    # Calculate total batches needed to process all images
    num_batches = (total_images + batch_size - 1) // batch_size
    print(f"\nProcessing {total_images} images with batch size {batch_size}")
    print(f"Total batches to process: {num_batches}")
    
    total_time = 0
    total_images_processed = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = images[start_idx:end_idx]
        actual_batch_size = len(batch_images)
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing images {start_idx} to {end_idx-1} ({actual_batch_size} images)")
        
        vllm_inputs = [
            {
                "prompt": text,
                "multi_modal_data": {
                    "image": [image]
                }
            } for image in batch_images
        ]
        
        start = time.perf_counter()
        results_batch = model.generate(prompts=vllm_inputs, sampling_params=sampling_params)
        torch.xpu.synchronize()
        end = time.perf_counter()
        
        batch_time = end - start
        total_time += batch_time
        total_images_processed += actual_batch_size
        
        batch_throughput = actual_batch_size / batch_time
        print(f"Batch time: {batch_time:.2f}s, Batch throughput: {batch_throughput:.2f} images/s")
        
        del vllm_inputs, results_batch
        torch.xpu.empty_cache()
    
    # Calculate overall statistics
    overall_throughput = total_images_processed / total_time
    avg_time_per_image = total_time / total_images_processed
    overall_iterations_per_second = num_batches / total_time
    
    print(f"\n{'='*60}")
    print(f"Overall Results:")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Overall throughput: {overall_iterations_per_second:.4f} batches/s, {overall_throughput:.2f} images/s")
    print(f"Average time per image: {avg_time_per_image:.4f}s")
    print(f"{'='*60}")
    '''
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
    #del model, vllm_inputs
    '''
    del model

if __name__ == "__main__":
    args = common_args.parse_args()
    run_qwen2_5_vl(args)

    

