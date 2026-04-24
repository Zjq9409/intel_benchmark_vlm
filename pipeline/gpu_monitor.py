"""
GPU 显存监控模块

支持：
  - NVIDIA GPU：nvidia-smi
  - Intel GPU ：xpu-smi（需要 sudo 免密权限）
"""

import json
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional

from loguru import logger

def _shorten_gpu_name(full_name: str) -> str:
    """将 GPU 完整名称缩短，如 NVIDIA GeForce RTX 4090 D → 4090D。
    也接受已用下划线替换空格的版本，如 NVIDIA_GeForce_RTX_4090_D → 4090D。"""
    import re as _re
    # 先将下划线还原为空格，统一处理
    name = full_name.replace('_', ' ')
    name = _re.sub(r'\b(NVIDIA|GeForce|Intel|Arc|Iris|Xe|UHD|RTX|GTX|RX|Radeon)\b', '', name, flags=_re.IGNORECASE)
    return _re.sub(r'\s+', '', name.strip()) or full_name.replace(' ', '').replace('_', '')


_XPU_METRICS = "0,1,2,3,18,22,24"
# 列索引（对应指标 0=util, 1=power, 2=freq, 3=temp, 18=vram, 22=vram_bw, 24=eu_util）
_XPU_COL_UTIL = 0
_XPU_COL_VRAM = 4


# ---------------------------------------------------------------------------
# 单次查询
# ---------------------------------------------------------------------------

def query_nvidia_sample(device_id: str = "0") -> Dict:
    """查询 NVIDIA GPU 一次采样：显存占用(MB) + GPU利用率(%)。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", device_id,
             "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.strip().split(",")]
            return {
                "vram_used_mb":  float(p[0]),
                "vram_total_mb": float(p[1]),
                "util_pct":      float(p[2]),
                "power_w":       float(p[3]),
                "temp_c":        float(p[4]),
            }
    except Exception:
        pass
    return {}


def query_intel_sample(device_id: str = "0") -> Dict:
    """查询 Intel GPU 一次采样（xpu-smi）。"""
    try:
        r = subprocess.run(
            ["sudo", "-n", "xpu-smi", "dump", _XPU_METRICS, "-d", str(device_id)],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            logger.debug(f"xpu-smi 失败: {r.stderr.strip()}")
            return {}
        lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
        for line in reversed(lines):
            parts = re.split(r",\s*", line)
            try:
                vals = [float(p) for p in parts]
                return {
                    "util_pct":     vals[0],
                    "power_w":      vals[1],
                    "freq_mhz":     vals[2],
                    "temp_c":       vals[3],
                    "vram_used_mb": vals[4],
                    "vram_bw_pct":  vals[5],
                    "eu_util_pct":  vals[6],
                }
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.debug(f"Intel sample 查询异常: {e}")
    return {}


def query_gpu_name(hwaccel: str, hwaccel_device: str = None) -> str:
    """查询 GPU 型号名称，用于文件名和图表标题。"""
    try:
        if hwaccel == "cuda":
            dev = hwaccel_device or "0"
            r = subprocess.run(
                ["nvidia-smi", "-i", dev, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return r.stdout.strip().replace(" ", "_")
        elif hwaccel in ("vaapi", "qsv"):
            dev = hwaccel_device or "/dev/dri/renderD128"
            r = subprocess.run(
                ["sudo", "-n", "xpu-smi", "discovery", "-d", "0"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                if "Device Name" in line or "name" in line.lower():
                    return line.split(":")[-1].strip().replace(" ", "_")
    except Exception:
        pass
    return "unknown"


def query_sample(hwaccel: str, hwaccel_device: Optional[str] = None) -> Dict:
    """根据 hwaccel 类型自动路由查询，返回当前 GPU 状态 dict。"""
    if hwaccel == "cuda":
        return query_nvidia_sample(hwaccel_device or "0")
    elif hwaccel in ("vaapi", "qsv"):
        dev = hwaccel_device or "0"
        m = re.search(r"renderD(\d+)", dev)
        if m:
            dev = str(int(m.group(1)) - 128)
        return query_intel_sample(dev)
    return {}


# ---------------------------------------------------------------------------
# 后台采样器
# ---------------------------------------------------------------------------

class GpuMemSampler:
    """后台线程持续采样 GPU 状态，统计峰值/均值，支持保存到 JSON。

    with 用法：
        with GpuMemSampler("cuda", "0") as s:
            run_ffmpeg(...)
        s.log_result()
        s.save_result("/out/gpu_stats.json")

    手动用法：
        s = GpuMemSampler("vaapi", "/dev/dri/renderD128").start()
        run_ffmpeg(...)
        s.stop()
        s.log_result()
        s.save_result("/out/gpu_stats.json")
    """

    def __init__(
        self,
        hwaccel: str,
        hwaccel_device: Optional[str] = None,
        interval: float = 0.3,
    ):
        self._hwaccel = hwaccel
        self._hwaccel_device = hwaccel_device
        self._interval = interval
        self._gpu_name = query_gpu_name(hwaccel, hwaccel_device)
        self._stop_event = threading.Event()
        self._samples: List[Dict] = []
        self._ts_start: float = 0.0
        self._ts_end: float = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stats: Dict = {}

    def __enter__(self) -> "GpuMemSampler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self) -> "GpuMemSampler":
        self._ts_start = time.time()
        self._thread.start()
        return self

    def stop(self) -> Dict:
        self._stop_event.set()
        self._thread.join()
        self._ts_end = time.time()

        def _agg(key: str) -> Dict:
            vals = [s[key] for s in self._samples if key in s and s[key] >= 0]
            if not vals:
                return {"peak": -1.0, "avg": -1.0}
            return {"peak": round(max(vals), 1), "avg": round(sum(vals) / len(vals), 1)}

        self.stats = {
            "hwaccel":         self._hwaccel,
            "hwaccel_device":  self._hwaccel_device,
            "gpu_name":        self._gpu_name,
            "duration_s":      round(self._ts_end - self._ts_start, 2),
            "sample_count":    len(self._samples),
            "sample_interval_s": self._interval,
            "vram_used_mb":    _agg("vram_used_mb"),
            "util_pct":        _agg("util_pct"),
        }
        # 可选字段（nvidia 专有）
        for key in ("power_w", "temp_c", "vram_total_mb"):
            agg = _agg(key)
            if agg["peak"] >= 0:
                self.stats[key] = agg
        # Intel 专有
        for key in ("eu_util_pct", "vram_bw_pct", "freq_mhz"):
            agg = _agg(key)
            if agg["peak"] >= 0:
                self.stats[key] = agg

        return self.stats

    def log_result(self) -> None:
        """以 INFO 级别打印采样结果。"""
        s = self.stats
        if not s:
            return
        vram = s.get("vram_used_mb", {})
        util = s.get("util_pct", {})
        if vram.get("peak", -1) >= 0:
            logger.info(
                f"GPU 解码监控  "
                f"显存: 峰值 {vram['peak']:.0f} MB / 均值 {vram['avg']:.0f} MB  |  "
                f"利用率: 峰值 {util.get('peak', -1):.0f}% / 均值 {util.get('avg', -1):.0f}%  |  "
                f"采样 {s['sample_count']} 次 / {s['duration_s']:.1f}s"
            )
        else:
            logger.info(f"GPU 解码监控: 查询不支持（hwaccel={self._hwaccel}）")

    def save_result(self, path: str) -> None:
        """将完整采样统计写入 JSON 文件。"""
        if not self.stats:
            return
        try:
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            logger.info(f"GPU 监控数据已保存: {path}")
        except Exception as e:
            logger.warning(f"GPU 监控数据保存失败: {e}")

    def save_plot(self, path: str, title: str = "") -> None:
        """将原始采样数据画成折线图并保存为 PNG。"""
        if not self._samples:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ts   = [s["timestamp_s"]  for s in self._samples]
            vram = [s.get("vram_used_mb", 0)  for s in self._samples]
            util = [s.get("util_pct", 0)      for s in self._samples]
            power= [s.get("power_w", None)    for s in self._samples]
            temp = [s.get("temp_c", None)     for s in self._samples]

            has_power = any(v is not None for v in power)
            has_temp  = any(v is not None for v in temp)
            n_rows = 2 + int(has_power) + int(has_temp)

            fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
            fig.suptitle(title or "GPU Decode Monitor", fontsize=11)

            row = 0
            axes[row].plot(ts, vram, color="#1f77b4", linewidth=1.5)
            axes[row].fill_between(ts, vram, alpha=0.2, color="#1f77b4")
            axes[row].set_ylabel("VRAM Used (MB)")
            axes[row].grid(True, linestyle="--", alpha=0.5)
            row += 1

            axes[row].plot(ts, util, color="#ff7f0e", linewidth=1.5)
            axes[row].fill_between(ts, util, alpha=0.2, color="#ff7f0e")
            axes[row].set_ylabel("GPU Util (%)")
            axes[row].set_ylim(0, 100)
            axes[row].grid(True, linestyle="--", alpha=0.5)
            row += 1

            if has_power:
                axes[row].plot(ts, power, color="#2ca02c", linewidth=1.5)
                axes[row].set_ylabel("Power (W)")
                axes[row].grid(True, linestyle="--", alpha=0.5)
                row += 1

            if has_temp:
                axes[row].plot(ts, temp, color="#d62728", linewidth=1.5)
                axes[row].set_ylabel("Temp (°C)")
                axes[row].grid(True, linestyle="--", alpha=0.5)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout()

            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            plt.savefig(path, dpi=120)
            plt.close(fig)
            logger.info(f"GPU 监控图表已保存: {path}")
        except Exception as e:
            logger.warning(f"GPU 图表保存失败: {e}")

    def save_samples_csv(self, path: str) -> None:
        """将原始采样序列保存为 CSV，便于用 pandas/matplotlib 画图。"""
        if not self._samples:
            return
        try:
            import csv, os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fieldnames = list(self._samples[0].keys())
            # 将 hwaccel_device 移到第一列
            if "hwaccel_device" in fieldnames:
                fieldnames.insert(0, fieldnames.pop(fieldnames.index("hwaccel_device")))
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._samples)
            logger.info(f"GPU 原始采样已保存: {path}  ({len(self._samples)} 行)")
        except Exception as e:
            logger.warning(f"GPU 原始采样保存失败: {e}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ts = time.time() - self._ts_start
            sample = query_sample(self._hwaccel, self._hwaccel_device)
            if sample:
                sample["hwaccel"] = self._hwaccel
                sample["hwaccel_device"] = self._hwaccel_device or "0"
                sample["timestamp_s"] = round(ts, 3)
                self._samples.append(sample)
            self._stop_event.wait(self._interval)
