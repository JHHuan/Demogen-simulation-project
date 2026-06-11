"""
从已保存的 *_1cam.pkl 和 *_2cam.pkl 生成论文用点云对比图。

支持两种用法:
1. 单个数据对:
   python compare_1cam_2cam_pointcloud.py --one path/to/demo_001_1cam.pkl

2. 批量处理整个目录:
   python compare_1cam_2cam_pointcloud.py --input-dir path/to/collected_data/20260524_123456

输出文件:
- *_pc_compare.png : RGB / Depth / 单相机点云 / 双相机点云对比图
- *_pc_1cam.png    : 单相机点云图
- *_pc_2cam.png    : 双相机点云图
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

AXIS_LABEL_FONTSIZE = 20
AXIS_TICK_FONTSIZE = 20
AXIS_LABEL_PAD = 12
AXIS_TICK_PAD = 6
AXIS_MAX_TICKS = 4


def _normalize_image_layout(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _convert_bgr_to_rgb(image):
    if image.ndim == 3 and image.shape[-1] == 3:
        return image[:, :, ::-1]
    return image


def _pc_colors_for_plot(pc, color_order):
    if pc.shape[1] < 6:
        return "#4C78A8"
    if color_order == "bgr":
        return np.clip(pc[:, [5, 4, 3]], 0.0, 1.0)
    return np.clip(pc[:, 3:6], 0.0, 1.0)


def _compute_bounds(pc_1cam, pc_2cam):
    xyz_list = []
    if len(pc_1cam) > 0:
        xyz_list.append(pc_1cam[:, :3])
    if len(pc_2cam) > 0:
        xyz_list.append(pc_2cam[:, :3])

    if not xyz_list:
        xyz_min = np.array([-0.2, -0.2, 0.3], dtype=np.float32)
        xyz_max = np.array([0.8, 0.2, 0.8], dtype=np.float32)
        return xyz_min, xyz_max

    xyz = np.concatenate(xyz_list, axis=0)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    pad = np.maximum((xyz_max - xyz_min) * 0.08, 0.02)
    return xyz_min - pad, xyz_max + pad


def _set_shared_axis_limits(ax, xyz_min, xyz_max):
    center = (xyz_min + xyz_max) / 2.0
    radius = max(np.max(xyz_max - xyz_min) / 2.0, 1e-3)
    ax.set_xlim(center[1] - radius, center[1] + radius)
    ax.set_ylim(center[0] - radius, center[0] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _plot_point_cloud(ax, pc, title, xyz_min, xyz_max, color_order):
    if len(pc) == 0:
        ax.set_title(f"{title}\n(empty)", fontsize=10)
        return

    ax.scatter(
        pc[:, 1],
        pc[:, 0],
        pc[:, 2],
        c=_pc_colors_for_plot(pc, color_order),
        s=1.2,
        depthshade=False,
    )
    ax.set_title(f"{title}\n({len(pc)} pts)", fontsize=10)
    ax.set_xlabel("Y", fontsize=AXIS_LABEL_FONTSIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("X", fontsize=AXIS_LABEL_FONTSIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Z", fontsize=AXIS_LABEL_FONTSIZE, labelpad=AXIS_LABEL_PAD)
    ax.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE, pad=AXIS_TICK_PAD)
    ax.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE, pad=AXIS_TICK_PAD)
    ax.tick_params(axis="z", labelsize=AXIS_TICK_FONTSIZE, pad=AXIS_TICK_PAD)
    ax.xaxis.set_major_locator(MaxNLocator(AXIS_MAX_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(AXIS_MAX_TICKS))
    ax.zaxis.set_major_locator(MaxNLocator(AXIS_MAX_TICKS))
    _set_shared_axis_limits(ax, xyz_min, xyz_max)
    ax.view_init(elev=24, azim=132)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_pair_from_one_cam_path(one_cam_path):
    if not one_cam_path.endswith("_1cam.pkl"):
        raise ValueError(f"文件名需要以 _1cam.pkl 结尾: {one_cam_path}")
    two_cam_path = one_cam_path.replace("_1cam.pkl", "_2cam.pkl")
    if not os.path.exists(two_cam_path):
        raise FileNotFoundError(f"未找到对应双相机文件: {two_cam_path}")
    return one_cam_path, two_cam_path


def _select_frame_index(n_frames, frame_idx):
    frame_idx = n_frames + frame_idx if frame_idx < 0 else frame_idx
    return max(0, min(frame_idx, n_frames - 1))


def save_pair_visuals(one_cam_path, two_cam_path, outdir=None, frame_idx=-1,
                      image_color_order="bgr", pc_color_order="bgr"):
    data_1cam = _load_pickle(one_cam_path)
    data_2cam = _load_pickle(two_cam_path)

    pc_1cam_all = np.asarray(data_1cam["point_cloud"])
    pc_2cam_all = np.asarray(data_2cam["point_cloud"])
    n_frames = min(len(pc_1cam_all), len(pc_2cam_all))
    if n_frames == 0:
        raise ValueError(f"空轨迹: {one_cam_path}")

    frame_idx = _select_frame_index(n_frames, frame_idx)
    pc_1cam = pc_1cam_all[frame_idx]
    pc_2cam = pc_2cam_all[frame_idx]

    image = _normalize_image_layout(np.asarray(data_1cam["image"])[frame_idx])
    if image_color_order == "bgr":
        image = _convert_bgr_to_rgb(image)
    depth = np.asarray(data_1cam["depth"])[frame_idx]
    xyz_min, xyz_max = _compute_bounds(pc_1cam, pc_2cam)

    prefix = os.path.basename(one_cam_path).replace("_1cam.pkl", "")
    output_dir = outdir or os.path.dirname(one_cam_path)
    os.makedirs(output_dir, exist_ok=True)

    compare_path = os.path.join(output_dir, f"{prefix}_pc_compare.png")
    single_path = os.path.join(output_dir, f"{prefix}_pc_1cam.png")
    dual_path = os.path.join(output_dir, f"{prefix}_pc_2cam.png")

    fig = plt.figure(figsize=(19, 5.8))
    ax_rgb = fig.add_subplot(1, 4, 1)
    ax_depth = fig.add_subplot(1, 4, 2)
    ax_pc_1cam = fig.add_subplot(1, 4, 3, projection="3d")
    ax_pc_2cam = fig.add_subplot(1, 4, 4, projection="3d")

    ax_rgb.imshow(image.astype(np.uint8))
    ax_rgb.set_title("Front RGB", fontsize=10)
    ax_rgb.axis("off")

    depth_im = ax_depth.imshow(depth, cmap="plasma")
    ax_depth.set_title("Front Depth", fontsize=10)
    ax_depth.axis("off")
    fig.colorbar(depth_im, ax=ax_depth, fraction=0.046, pad=0.04)

    _plot_point_cloud(ax_pc_1cam, pc_1cam, "Single Camera", xyz_min, xyz_max, pc_color_order)
    _plot_point_cloud(ax_pc_2cam, pc_2cam, "Dual Camera", xyz_min, xyz_max, pc_color_order)

    fig.suptitle(f"{prefix}  frame {frame_idx + 1}/{n_frames}", fontsize=12)
    fig.tight_layout()
    fig.savefig(compare_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for path, pc, title in [
        (single_path, pc_1cam, "Single Camera Point Cloud"),
        (dual_path, pc_2cam, "Dual Camera Point Cloud"),
    ]:
        fig_single = plt.figure(figsize=(7.6, 7.2))
        ax_single = fig_single.add_subplot(111, projection="3d")
        _plot_point_cloud(ax_single, pc, title, xyz_min, xyz_max, pc_color_order)
        fig_single.tight_layout()
        fig_single.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig_single)

    print(f"Saved: {compare_path}")
    print(f"Saved: {single_path}")
    print(f"Saved: {dual_path}")


def _iter_one_cam_files(input_dir):
    for name in sorted(os.listdir(input_dir)):
        if name.endswith("_1cam.pkl"):
            yield os.path.join(input_dir, name)


def main():
    parser = argparse.ArgumentParser(description="生成单相机/双相机点云对比图")
    parser.add_argument("--one", type=str, default=None,
                        help="单相机 pkl 路径，脚本会自动匹配对应的 *_2cam.pkl")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="批量处理目录，自动查找其中全部 *_1cam.pkl")
    parser.add_argument("--outdir", type=str, default=None,
                        help="图片输出目录，默认保存到 pkl 同目录")
    parser.add_argument("--frame-idx", type=int, default=-1,
                        help="可视化第几帧，默认 -1 表示最后一帧")
    parser.add_argument("--image-color-order", choices=["bgr", "rgb"], default="bgr",
                        help="图像通道顺序，当前采集脚本默认更接近 bgr")
    parser.add_argument("--pc-color-order", choices=["bgr", "rgb"], default="bgr",
                        help="点云颜色通道顺序，当前采集脚本默认更接近 bgr")
    args = parser.parse_args()

    if bool(args.one) == bool(args.input_dir):
        raise ValueError("请二选一传入 --one 或 --input-dir")

    if args.one:
        one_cam_path, two_cam_path = _resolve_pair_from_one_cam_path(args.one)
        save_pair_visuals(
            one_cam_path,
            two_cam_path,
            outdir=args.outdir,
            frame_idx=args.frame_idx,
            image_color_order=args.image_color_order,
            pc_color_order=args.pc_color_order,
        )
        return

    one_cam_files = list(_iter_one_cam_files(args.input_dir))
    if not one_cam_files:
        raise FileNotFoundError(f"目录中未找到 *_1cam.pkl: {args.input_dir}")

    for one_cam_path in one_cam_files:
        one_cam_path, two_cam_path = _resolve_pair_from_one_cam_path(one_cam_path)
        save_pair_visuals(
            one_cam_path,
            two_cam_path,
            outdir=args.outdir,
            frame_idx=args.frame_idx,
            image_color_order=args.image_color_order,
            pc_color_order=args.pc_color_order,
        )


if __name__ == "__main__":
    main()
