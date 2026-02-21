#!/usr/bin/env python3
"""
交互式 SAM 分割脚本
使用点击来选择要分割的物体
cd /home/hjh/git_code/demogen/DemoGen-master/data/sam_mask
python segment_interactive.py \
    --image "0218-cube/0/source.jpg" \
    --output "0218-cube/0/green cube.jpg"

"""

import os
import numpy as np
from PIL import Image
import torch
import cv2


def segment_interactive(image_path, output_path):
    """
    交互式分割：显示图像，让用户点击选择物体

    Args:
        image_path: 输入图像路径
        output_path: 输出mask路径
    """
    from segment_anything import sam_model_registry, SamPredictor

    # 权重目录
    weights_dir = "/home/hjh/git_code/demogen/DemoGen-master/data/sam_mask/weights"
    os.makedirs(weights_dir, exist_ok=True)

    # 可用的模型
    checkpoints = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }

    # 检查权重文件是否存在
    model_type = "vit_b"  # 使用base模型（速度最快）
    checkpoint_filename = os.path.join(weights_dir, os.path.basename(checkpoints[model_type]))

    if not os.path.exists(checkpoint_filename):
        print(f"权重文件不存在: {checkpoint_filename}")
        print(f"正在下载 {model_type} 权重...")
        import urllib.request
        urllib.request.urlretrieve(checkpoints[model_type], checkpoint_filename)
        print(f"下载完成: {checkpoint_filename}")

    print(f"加载 SAM 模型 ({model_type})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_filename)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    # 显示图像，获取点击
    print("\n" + "="*60)
    print("交互式分割模式")
    print("="*60)
    print("操作说明:")
    print("  - 鼠标左键点击: 选择前景点（绿色圆点）")
    print("  - 鼠标右键点击: 选择背景点（红色圆点）")
    print("  - 按 's' 键: 执行分割并显示结果")
    print("  - 按 'c' 键: 清除所有点")
    print("  - 按 'q' 键: 保存mask并退出")
    print("="*60 + "\n")

    point_coords = []
    point_labels = []

    clone = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键：前景点
            point_coords.append([x, y])
            point_labels.append(1)
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", param)
            print(f"添加前景点: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键：背景点
            point_coords.append([x, y])
            point_labels.append(0)
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", param)
            print(f"添加背景点: ({x}, {y})")

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", on_mouse, clone)

    mask_result = None

    while True:
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 保存并退出
            if mask_result is not None:
                # 保存mask为二值图像（0和255）
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                mask_image = Image.fromarray((mask_result * 255).astype(np.uint8))
                mask_image.save(output_path)
                print(f"\n✓ Mask已保存: {output_path}")
            else:
                print("\n警告：没有生成mask，按's'键先执行分割")
            break

        elif key == ord('c'):
            # 清除所有点
            point_coords.clear()
            point_labels.clear()
            clone = image.copy()
            cv2.imshow("Image", clone)
            print("已清除所有点")

        elif key == ord('s') and len(point_coords) > 0:
            # 执行分割
            print(f"\n执行分割（共 {len(point_coords)} 个点）...")

            input_point = np.array(point_coords)
            input_label = np.array(point_labels)

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # 选择得分最高的mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            print(f"最佳mask得分: {best_score:.3f}")

            # 保存结果
            mask_result = best_mask

            # 可视化结果
            result = image.copy()
            # 半透明叠加mask
            result[best_mask] = result[best_mask] * 0.5 + np.array([0, 255, 0]) * 0.5

            # 重新绘制所有点
            for i, (coord, label) in enumerate(zip(point_coords, point_labels)):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(result, tuple(coord), 5, color, -1)

            cv2.imshow("Result", result)
            print("按 'q' 保存mask并退出")

    cv2.destroyAllWindows()

    return mask_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="交互式SAM分割")
    parser.add_argument("--image", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--output", type=str,
                        default="/home/hjh/git_code/demogen/DemoGen-master/data/sam_mask/0216-cube/green cube.jpg",
                        help="输出mask路径")

    args = parser.parse_args()

    print("="*60)
    print("SAM 交互式分割工具")
    print("="*60)
    print(f"输入图像: {args.image}")
    print(f"输出mask: {args.output}")
    print("="*60 + "\n")

    try:
        mask = segment_interactive(args.image, args.output)
        if mask is not None:
            print("\n分割完成！")
        else:
            print("\n没有生成mask")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
