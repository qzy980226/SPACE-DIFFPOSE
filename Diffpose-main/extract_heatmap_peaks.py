"""
从检测器输出的 2D 关键点热力图中提取每个通道的最大响应值，输出 JSON 文件。

热力图格式假设：每个文件对应一张图像，数组形状为 [K, H, W]
  K = 关键点数量（本项目为 11）
  H, W = 热力图空间分辨率

用法示例：
  python extract_heatmap_peaks.py \
      --heatmap_dir ./data/heatmaps \
      --output ./heatmap_peaks.json \
      --format npz \
      --npz_key heatmaps
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


KEYPOINT_NAMES = [
    "kp_0", "kp_1", "kp_2", "kp_3", "kp_4",
    "kp_5", "kp_6", "kp_7", "kp_8", "kp_9", "kp_10",
]


def load_heatmap(filepath: Path, fmt: str, npz_key: str, h5_key: str) -> np.ndarray:
    """加载单个热力图文件，返回 ndarray [K, H, W]。"""
    if fmt == "npz":
        data = np.load(filepath)
        if npz_key not in data:
            available = list(data.keys())
            raise KeyError(
                f"{filepath.name}: 找不到 key '{npz_key}'，"
                f"可用 key：{available}。请用 --npz_key 指定。"
            )
        return data[npz_key].astype(np.float32)

    elif fmt == "npy":
        return np.load(filepath).astype(np.float32)

    elif fmt in ("h5", "hdf5"):
        import h5py
        with h5py.File(filepath, "r") as f:
            if h5_key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"{filepath.name}: 找不到 dataset '{h5_key}'，"
                    f"可用 key：{available}。请用 --h5_key 指定。"
                )
            return f[h5_key][()].astype(np.float32)

    else:
        raise ValueError(f"不支持的格式：{fmt}，请选择 npz / npy / h5")


def extract_peaks(heatmap: np.ndarray, n_kpts: int) -> dict:
    """
    从 [K, H, W] 热力图中提取每个关键点通道的最大响应值。
    返回 {keypoint_name: max_value} 字典。
    """
    if heatmap.ndim != 3:
        raise ValueError(f"热力图维度应为 3（[K,H,W]），实际为 {heatmap.ndim}。")
    if heatmap.shape[0] < n_kpts:
        raise ValueError(
            f"热力图通道数 {heatmap.shape[0]} 小于期望关键点数 {n_kpts}。"
        )

    result = {}
    for k in range(n_kpts):
        name = KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"kp_{k}"
        result[name] = float(heatmap[k].max())
    return result


def main():
    parser = argparse.ArgumentParser(description="提取热力图最大响应值 → JSON")
    parser.add_argument(
        "--heatmap_dir", required=True,
        help="热力图文件所在目录（每个文件对应一张图像）"
    )
    parser.add_argument(
        "--output", default="heatmap_peaks.json",
        help="输出 JSON 文件路径（默认：heatmap_peaks.json）"
    )
    parser.add_argument(
        "--format", default="npz", choices=["npz", "npy", "h5", "hdf5"],
        help="热力图文件格式（默认：npz）"
    )
    parser.add_argument(
        "--npz_key", default="heatmaps",
        help="npz 格式中热力图数组的 key（默认：heatmaps）"
    )
    parser.add_argument(
        "--h5_key", default="heatmaps",
        help="h5 格式中热力图 dataset 的 key（默认：heatmaps）"
    )
    parser.add_argument(
        "--n_kpts", type=int, default=11,
        help="关键点数量（默认：11，对应 SPEED+ 航天器）"
    )
    parser.add_argument(
        "--ext", default=None,
        help="强制指定文件扩展名（如 .npz），不指定则按 --format 自动推断"
    )
    args = parser.parse_args()

    heatmap_dir = Path(args.heatmap_dir)
    if not heatmap_dir.is_dir():
        raise FileNotFoundError(f"目录不存在：{heatmap_dir}")

    # 确定文件扩展名
    ext = args.ext if args.ext else f".{args.format.split('/')[-1]}"
    if not ext.startswith("."):
        ext = "." + ext

    files = sorted(heatmap_dir.glob(f"*{ext}"))
    if not files:
        raise FileNotFoundError(
            f"在 {heatmap_dir} 中未找到 '{ext}' 文件，"
            f"请检查 --heatmap_dir 和 --format 参数。"
        )

    print(f"找到 {len(files)} 个热力图文件，开始提取...")

    results = {}
    errors = []

    for i, fpath in enumerate(files):
        image_key = fpath.stem          # e.g. "img000001" or "img000001_gmm_params"
        try:
            heatmap = load_heatmap(fpath, args.format, args.npz_key, args.h5_key)
            results[image_key] = extract_peaks(heatmap, args.n_kpts)
        except Exception as e:
            errors.append(f"  [{fpath.name}] {e}")
            continue

        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"  已处理 {i + 1}/{len(files)}")

    # 输出 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n完成：{len(results)} 张图像写入 {output_path}")

    if errors:
        print(f"\n以下 {len(errors)} 个文件处理失败：")
        for msg in errors:
            print(msg)


if __name__ == "__main__":
    main()
