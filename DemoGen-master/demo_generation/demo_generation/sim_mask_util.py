"""
仿真环境专用的mask处理工具
参考mask_util.py，但适配MuJoCo仿真点云
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_camera_intrinsics(height, width, fovy_deg):
    """
    获取相机内参矩阵
    """
    import math
    fovy = math.radians(fovy_deg)
    f = height / (2 * math.tan(fovy / 2))
    cx = width / 2
    cy = height / 2

    K = np.array([
        [f,  0, cx],
        [0, f, cy],
        [0, 0,  1]
    ])

    return K


def project_points_to_image(points_3d, K, image_size):
    """
    将3D点投影到2D图像

    Args:
        points_3d: (N, 3) 世界坐标系中的点
        K: 相机内参矩阵 (3, 3)
        image_size: (height, width)

    Returns:
        pixel_coords: (N, 2) 像素坐标
    """
    # 简化：对于仿真环境，点云已经在世界坐标系
    # 我们需要将世界坐标转换到相机坐标

    # 相机参数（从grasping_demogen.xml）
    camera_pos = np.array([1.0, 0.0, 0.7])
    camera_quat = np.array([0.56, 0.43, 0.43, 0.56])  # [w, x, y, z]

    # 转换为scipy格式
    camera_quat_scipy = np.array([0.43, 0.43, 0.56, 0.56])  # [x, y, z, w]
    cam_rot = R.from_quat(camera_quat_scipy).as_matrix()

    # 世界坐标 -> 相机坐标
    points_cam = (points_3d - camera_pos) @ cam_rot.T

    # 投影到图像平面
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_proj = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    y_proj = fy * points_cam[:, 1] / points_cam[:, 2] + cy

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


def get_objects_pcd_from_sam_mask_sim(point_cloud_robot, mask, depth_shape=(240, 320)):
    """
    仿真环境版本：从SAM mask中提取物体点云

    Args:
        point_cloud_robot: 机器人坐标系中的点云 (N, 6) XYZ+RGB
        mask: SAM分割mask (H, W)
        depth_shape: 深度图尺寸 (height, width)

    Returns:
        物体的点云 (M, 6) XYZ+RGB
    """
    # 提取XYZ和RGB
    points_xyz = point_cloud_robot[:, :3]
    points_rgb = point_cloud_robot[:, 3:]

    # 获取相机内参
    K = get_camera_intrinsics(depth_shape[0], depth_shape[1], fovy_deg=45)
    image_size = depth_shape

    # 用mask过滤点
    filtered_xyz = filter_points_by_mask(points_xyz, mask, K, image_size)

    if len(filtered_xyz) == 0:
        print(f"警告：mask过滤后没有点！点云范围: {np.min(points_xyz, axis=0)} ~ {np.max(points_xyz, axis=0)}")
        return point_cloud_robot  # 返回原始点云

    # 找到对应的RGB值
    # 由于mask过滤，需要重新匹配RGB
    # 简化方法：使用最近邻或直接保留所有RGB
    # 更精确的方法需要记录索引

    # 简化：对所有RGB，只保留对应XYZ的RGB
    # 这里使用距离最近的方法
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

    print(f"原始点云: {len(points_xyz)} 点")
    print(f"过滤后点云: {len(filtered_xyz)} 点")

    return filtered_pcd


if __name__ == "__main__":
    # 测试
    print("仿真环境Mask处理工具")

    # 测试投影
    test_points = np.array([[0.3, 0.0, 0.45]])  # 工作空间中心附近
    K = get_camera_intrinsics(240, 320, 45)
    pixel_coords = project_points_to_image(test_points, K, (240, 320))
    print(f"测试点投影: {pixel_coords}")
