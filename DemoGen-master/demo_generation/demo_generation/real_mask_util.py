"""
真实环境专用的mask处理工具
适配RealSense L515相机和真实点云数据
"""
import numpy as np
from demo_generation.real_camera import CAMERA_TO_WORLD_TRANSFORM, CAMERA_INTRINSICS, DEPTH_SCALE


def get_camera_intrinsics(target_width=None, target_height=None):
    """
    获取RealSense L515相机内参矩阵

    Args:
        target_width: 目标图像宽度（如果提供，会缩放内参）
        target_height: 目标图像高度（如果提供，会缩放内参）

    Returns:
        K: 3x3相机内参矩阵
    """
    # 基础内参（RGB相机原始分辨率1280x720）
    base_width = CAMERA_INTRINSICS['width']
    base_height = CAMERA_INTRINSICS['height']
    base_fx = CAMERA_INTRINSICS['fx']
    base_fy = CAMERA_INTRINSICS['fy']
    base_cx = CAMERA_INTRINSICS['cx']
    base_cy = CAMERA_INTRINSICS['cy']

    # 如果提供了目标尺寸，缩放内参
    if target_width is not None and target_height is not None:
        scale_x = target_width / base_width
        scale_y = target_height / base_height
        fx = base_fx * scale_x
        fy = base_fy * scale_y
        cx = base_cx * scale_x
        cy = base_cy * scale_y
    else:
        fx = base_fx
        fy = base_fy
        cx = base_cx
        cy = base_cy

    K = np.array([
        [fx,  0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ])

    return K


def project_points_to_image(points_3d, K, image_size):
    """
    将3D点投影到2D图像（与数据采集时的点云生成逻辑一致）

    真实环境使用的是RealSense相机的坐标变换，与仿真不同。

    Args:
        points_3d: (N, 3) 世界坐标系中的点
        K: 相机内参矩阵 (3, 3)
        image_size: (height, width)

    Returns:
        pixel_coords: (N, 2) 像素坐标
    """
    # 真实环境：相机坐标系到世界坐标系的变换是CAMERA_TO_WORLD_TRANSFORM
    # 反向投影：世界坐标 -> 相机坐标需要用逆变换

    T_world_to_camera = np.linalg.inv(CAMERA_TO_WORLD_TRANSFORM)

    # 1. 世界坐标 -> 相机坐标系
    # 齐次坐标
    points_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    points_cam_homo = (T_world_to_camera @ points_homo.T).T
    points_cam = points_cam_homo[:, :3]

    # 2. 投影到图像平面
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 相机坐标系中，Z是深度方向
    # 避免除零
    z = points_cam[:, 2]
    valid_z = np.abs(z) > 1e-6

    x_proj = np.zeros_like(z)
    y_proj = np.zeros_like(z)

    x_proj[valid_z] = fx * points_cam[valid_z, 0] / z[valid_z] + cx
    y_proj[valid_z] = fy * points_cam[valid_z, 1] / z[valid_z] + cy

    pixel_coords = np.stack([x_proj, y_proj], axis=1)

    return pixel_coords


def filter_points_by_mask(points_3d, mask, K, image_size):
    """
    根据mask过滤3D点云

    Args:
        points_3d: (N, 3) 世界坐标系中的点
        mask: (H, W) 二值mask
        K: 相机内参
        image_size: (height, width)

    Returns:
        filtered_points: (M, 3) 过滤后的点
    """
    # 投影到图像
    pixel_coords = project_points_to_image(points_3d, K, image_size)

    # 检查是否在图像范围内
    height, width = image_size
    valid_pixels = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
    )

    # 对于在图像范围内的点，检查mask值
    mask_values = np.zeros(len(points_3d), dtype=bool)

    if np.any(valid_pixels):
        # 获取mask值（注意：mask的坐标系是(row, col) = (y, x)）
        mask_values[valid_pixels] = mask[
            pixel_coords[valid_pixels, 1].astype(int),
            pixel_coords[valid_pixels, 0].astype(int)
        ]

    # 过滤点
    filtered_points = points_3d[mask_values]

    return filtered_points


def get_objects_pcd_from_sam_mask_real(point_cloud_robot, mask, depth_shape=None):
    """
    真实环境版本：从SAM mask中提取物体点云

    Args:
        point_cloud_robot: 世界坐标系中的点云 (N, 6) XYZ+RGB
        mask: SAM分割mask (H, W)
        depth_shape: 深度图尺寸 (height, width)，已弃用，保留用于兼容性

    Returns:
        物体的点云 (M, 6) XYZ+RGB
    """
    # 提取XYZ和RGB
    points_xyz = point_cloud_robot[:, :3]
    points_rgb = point_cloud_robot[:, 3:]

    # 使用mask的实际尺寸
    mask_height, mask_width = mask.shape[:2]

    # 获取适配mask分辨率的相机内参
    K = get_camera_intrinsics(target_width=mask_width, target_height=mask_height)
    image_size = (mask_height, mask_width)

    # 用mask过滤点
    filtered_xyz = filter_points_by_mask(points_xyz, mask, K, image_size)

    if len(filtered_xyz) == 0:
        return point_cloud_robot  # 返回原始点云

    # 找到对应的RGB值
    # 使用距离最近的方法匹配RGB
    from scipy.spatial import cKDTree

    if len(filtered_xyz) < len(points_xyz):
        # 构建原始点的KD树
        tree = cKDTree(points_xyz)
        # 对于每个过滤后的点，找到最近的原始点的RGB
        _, indices = tree.query(filtered_xyz)
        filtered_rgb = points_rgb[indices]
    else:
        # 如果过滤后的点数相同或更多，直接使用
        filtered_rgb = points_rgb[:len(filtered_xyz)]

    # 合并XYZ和RGB
    filtered_pcd = np.concatenate([filtered_xyz, filtered_rgb], axis=1)

    return filtered_pcd


if __name__ == "__main__":
    # 测试
    print("真实环境Mask处理工具")

    # 测试不同分辨率的内参缩放
    print("\n测试内参缩放:")
    K_720p = get_camera_intrinsics(target_width=1280, target_height=720)
    print(f"720p内参:\nfx={K_720p[0,0]:.2f}, fy={K_720p[1,1]:.2f}, cx={K_720p[0,2]:.2f}, cy={K_720p[1,2]:.2f}")

    K_1080p = get_camera_intrinsics(target_width=1920, target_height=1080)
    print(f"1080p内参:\nfx={K_1080p[0,0]:.2f}, fy={K_1080p[1,1]:.2f}, cx={K_1080p[0,2]:.2f}, cy={K_1080p[1,2]:.2f}")

    # 测试投影
    test_points = np.array([[0.3, 0.0, 0.45]])  # 工作空间中心附近
    pixel_coords = project_points_to_image(test_points, K_720p, (720, 1280))
    print(f"\n测试点投影到720p图像: {pixel_coords}")
