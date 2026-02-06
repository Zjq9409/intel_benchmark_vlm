import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/hugging-face-llm/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--imgs-path", type=str, default="image/qwen_qoe_test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--multi-batch", action="store_true", help="Test multiple batch sizes (1,2,4,8,16,32,64,128)")
    parser.add_argument("--text", type=str, default="Describe this video")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument('--visual_engine_dir',
                        type=str,
                        default="/root/media_infra_llm/llava/to_nv/qwen2vl-7b_0.17.post1/vision_encoder",
                        help='Directory containing visual TRT engines')
    parser.add_argument('--visual_engine_name',
                        type=str,
                        default='model.engine',
                        help='Name of visual TRT engine')
    parser.add_argument('--llm_engine_dir',
                        type=str,
                        default="/root/media_infra_llm/llava/to_nv/qwen2vl-7b_0.17.post1/llm_engine",
                        help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf-model-dir',
                        type=str,
                        default="/root/hugging-face-llm/Qwen2-VL-7B-Instruct",
                        help="Directory containing tokenizer")
    parser.add_argument('--input-text',
                        type=str,
                        default="Describe this video",
                        help='Text prompt to LLM')

    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tp_size', type=list, default=[0])
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--run_profiling',
                        action='store_true',
                        help='Profile runtime over several iterations')
    parser.add_argument('--profiling_iterations',
                        type=int,
                        help="Number of iterations to run profiling",
                        default=20)
    parser.add_argument('--check_accuracy',
                        action='store_true',
                        help='Check correctness of text output')
    parser.add_argument("--path_sep",
                        type=str,
                        default=",",
                        help='Path separator symbol')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        action='store_true',
                        default=None,
                        help="Enable FMHA runner FP32 accumulation.")
    parser.add_argument(
        '--enable_chunked_context',
        action='store_true',
        help='Enables chunked context (only available with cpp session).',
    )
    parser.add_argument(
        '--use_py_session',
        default=False,
        action='store_true',
        help=
        "Whether or not to use Python runtime session. By default C++ runtime session is used for the LLM."
    )
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.5,
        type=float,
        help='Specify the free gpu memory fraction.',
    )
    parser.add_argument(
        '--cross_kv_cache_fraction',
        default=0.9,
        type=float,
        help=
        'Specify the kv cache fraction reserved for cross attention. Only applicable for encoder-decoder models. By default 0.5 for self and 0.5 for cross.',
    )
    parser.add_argument(
        '--multi_block_mode',
        type=lambda s: s.lower() in
        ("yes", "true", "t", "1"
         ),  # custom boolean function to convert input string to boolean
        default=True,
        help=
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel."
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Enforce eager mode for the LLM."
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="shapes for images resize. format: \"h1,w1;h2,w2;...\""
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="result.txt",
        help="result path for output."
    )
    parser.add_argument(
        "--result_txt_path",
        type=str,
        default="result.txt",
        help="result path for output."
    )
    parser.add_argument(
        "--system_prompt",
        action="store_true",
        help="system prompt for llm."
    )
    return parser.parse_args()