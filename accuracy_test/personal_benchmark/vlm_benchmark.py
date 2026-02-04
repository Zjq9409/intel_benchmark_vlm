import requests
import json
import random
import base64
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time
from transformers import PreTrainedTokenizerBase
from argparse import ArgumentParser as FlexibleArgumentParser
import argparse

import numpy as np
from PIL import Image
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

from utils import RequestFuncInput, RequestFuncOutput, BenchmarkMetrics, async_request_openai_chat_completions, calculate_metrics
from transformers import AutoTokenizer
 
# TARGET_SIZE = (100, 100)
# from io import BytesIO

# with Image.open(file_loc) as img:
#         img = img.resize(TARGET_SIZE) 
#         buffered = BytesIO()
#         img.save(buffered, format="PNG")
#         base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

async def benchmark(
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    ignore_eos: bool,
):
    tasks: List[asyncio.Task] = []
    test_prompt, test_prompt_len,test_output_len, test_mm_content,  = (
    input_requests[0])
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )
    test_output = await async_request_openai_chat_completions(request_func_input=test_input)
    if not ignore_eos:
       print("输出结果：",test_output.generated_text)
    benchmark_start_time = time.perf_counter()
    for request in input_requests:
        prompt, prompt_len, test_output_len, mm_content = request
        request_func_input = RequestFuncInput(model=model_id,
                                              model_name=model_name,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=test_output_len,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                async_request_openai_chat_completions(request_func_input=request_func_input,
                                     )))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total image and input tokens:", metrics.image_prompt_input_lens))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))
    
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        selected_percentile_metrics = ['ttft', 'tpot', 'itl']
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result



def main(args: argparse.Namespace):
    sampled_requests: List[Tuple[str, int, int, Dict[str,
                                                     Collection[str]]]] = []
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.model
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id,
                              trust_remote_code=True
                              )
    args = parser.parse_args()
    batch_size = args.batch_size
    image_path = args.image_path
    print("image size: ",Image.open(image_path).size)

    with open(image_path, 'rb') as file:
        image_data = file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
    sampled_requests: List[Tuple[str, int, int, dict]] = []
    for i in range(batch_size):
        mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            }
        texts = args.prompt
        prompt_token_ids = tokenizer(texts).input_ids
        prompt_len = len(prompt_token_ids)
        # print("input prompt len: ", prompt_len)
        output_len = args.output_len
        sampled_requests.append((texts, prompt_len, output_len, mm_content))

    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=sampled_requests,
            ignore_eos=args.ignore_eos,)
    )

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
            description="Benchmark the online serving throughput.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="concurrent request number",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=512,
        help="maximum output length",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="image file path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请描述这张图片",
        help="image prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="model path"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint.",
    )
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    args = parser.parse_args()
    main(args)