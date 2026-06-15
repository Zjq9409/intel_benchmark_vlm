import math
import sys
from PIL import Image
from transformers import AutoTokenizer

# ── 模型参数 (Qwen3-VL-2B) ──────────────────────────────────────────
MODEL_PATH     = "/DISK0/Qwen3.5-35B-A3B/"
PATCH_SIZE     = 16
TEMPORAL_PATCH = 2
MERGE_SIZE     = 2
SHORTEST_EDGE  = 65536
LONGEST_EDGE   = 16777216
FACTOR         = PATCH_SIZE * MERGE_SIZE  # = 32


def calc_visual_tokens(image_path=None, prompt="Describe this image.", tokenizer=None, size=None):
    # ── Step 0: 读取图片 / 直接使用尺寸 ───────────────────────────────────
    if size is not None:
        W, H = size
        print(f"Step 0  图片尺寸 (手动指定):  {W}x{H} (WxH),  {W*H:,} 像素")
    else:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        print(f"Step 0  读取图片:  {W}x{H} (WxH),  {W*H:,} 像素")

    # ── Step 1: smart_resize ─────────────────────────────────────────
    H_new = round(H / FACTOR) * FACTOR
    W_new = round(W / FACTOR) * FACTOR
    area  = H_new * W_new

    scale = 1.0
    if area < SHORTEST_EDGE:
        scale = math.sqrt(SHORTEST_EDGE / area)
        H_new = round(H_new * scale / FACTOR) * FACTOR
        W_new = round(W_new * scale / FACTOR) * FACTOR
        area  = H_new * W_new
    elif area > LONGEST_EDGE:
        scale = math.sqrt(LONGEST_EDGE / area)
        H_new = round(H_new * scale / FACTOR) * FACTOR
        W_new = round(W_new * scale / FACTOR) * FACTOR
        area  = H_new * W_new

    print(f"Step 1  smart_resize:")
    print(f"        H: {H} -> round({H}/{FACTOR})*{FACTOR} = {round(H/FACTOR)}*{FACTOR} = {H_new}")
    print(f"        W: {W} -> round({W}/{FACTOR})*{FACTOR} = {round(W/FACTOR)}*{FACTOR} = {W_new}")
    if scale != 1.0:
        print(f"        面积超出范围，缩放 x{scale:.4f}")
    else:
        print(f"        面积 {area:,} in [{SHORTEST_EDGE:,}, {LONGEST_EDGE:,}] -> 无需缩放")
    print(f"        -> resize {W}x{H} -> {W_new}x{H_new}")

    # ── Step 2: 切 patch ─────────────────────────────────────────────
    grid_h    = H_new // PATCH_SIZE
    grid_w    = W_new // PATCH_SIZE
    N_patches = grid_h * grid_w
    patch_dim = 3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE   # 1536
    print(f"Step 2  切 patch:")
    print(f"        grid_h = {H_new}/{PATCH_SIZE} = {grid_h}")
    print(f"        grid_w = {W_new}/{PATCH_SIZE} = {grid_w}")
    print(f"        N_patches = {grid_h}x{grid_w} = {N_patches:,}")
    print(f"        pixel_values shape = [{N_patches}, {patch_dim}]"
          f"  ({patch_dim} = 3x{TEMPORAL_PATCH}x{PATCH_SIZE}x{PATCH_SIZE})")

    # ── Step 3: PatchMerger 2x2 ──────────────────────────────────────
    N_visual = N_patches // (MERGE_SIZE ** 2)
    print(f"Step 3  PatchMerger {MERGE_SIZE}x{MERGE_SIZE}:")
    print(f"        N_visual_tokens = {N_patches} / {MERGE_SIZE**2} = {N_visual:,}")
    print(f"        image_grid_thw = (1, {grid_h}, {grid_w})")

    # ── Step 4: 组装 input_ids (用 tokenizer 精确计算文本 token) ────────
    #
    # chat 模板展开后结构（有图片时）:
    #   <|im_start|>user\n<|vision_start|>[N_visual*<|image_pad|>]<|vision_end|>\n{prompt}<|im_end|>\n
    #   <|im_start|>assistant\n
    #
    # 策略：用纯文本消息走 apply_chat_template（不含图片，因此不含 vision_start/end），
    #        再 +2 补上 <|vision_start|> 和 <|vision_end|>。
    tmpl_ids     = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids   = tokenizer.encode(prompt, add_special_tokens=False)
    struct_count = len(tmpl_ids) - len(prompt_ids)   # 纯结构 token
    text_tokens  = len(tmpl_ids) + 2                 # +2: vision_start + vision_end
    total        = N_visual + text_tokens

    decoded = tokenizer.decode(tmpl_ids, skip_special_tokens=False)
    print(f"Step 4  组装 input_ids (tokenizer 精确计算):")
    print(f"        chat 模板展开 (无图,仅文本):")
    print(f"          {repr(decoded)}")
    print(f"        模板结构 token:   {struct_count}  (im_start/end, user, assistant, \\n ...)")
    print(f"        prompt token:    {len(prompt_ids)}  ({repr(prompt)})")
    print(f"        vision_start + vision_end: 2")
    print(f"        文本合计:        {text_tokens}")
    print(f"        视觉 token:      {N_visual:,}")
    print(f"        ─────────────────────────────────────────")
    print(f"        总计:            {total:,}  -> torch.Size([1, {total}])")

    return total



if __name__ == "__main__":
    import re
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image path or WxH size")
    parser.add_argument("--prompt_path", help="Path to prompt file")
    args = parser.parse_args()
    
    arg = args.input
    prompt = "Describe this image."
    if args.prompt_path:
        with open(args.prompt_path, "r") as f:
            prompt = f.read().strip()

    # 判断参数是 WxH 尺寸 还是 文件路径
    m = re.fullmatch(r'(\d+)[xX](\d+)', arg)
    if m:
        use_size = (int(m.group(1)), int(m.group(2)))
        use_path = None
        label = f"尺寸: {arg}"
    else:
        use_size = None
        use_path = arg
        label = f"图片路径: {arg}"

    print(f"加载 tokenizer: {MODEL_PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"加载完成\n")
    print(f"{'='*55}")
    print(f"  {label}")
    print(f"  Prompt length: {len(prompt)}")
    print(f"{'='*55}\n")
    calc_visual_tokens(image_path=use_path, tokenizer=tok, size=use_size, prompt=prompt)
