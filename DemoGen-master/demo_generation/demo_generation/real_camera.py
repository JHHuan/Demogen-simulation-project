"""
真实环境相机参数配置
适配RealSense L515相机（用于FR3+DexHand数据采集）
"""
import numpy as np

################################# Camera Calibration ##############################################
# RealSense L515相机配置
# RGB: 1280x720, Depth: 640x480

# 相机外参变换矩阵（相机坐标系 -> 世界坐标系，含-2°滚转角调整）
# 来自数据采集脚本的真实标定参数
CAMERA_TO_WORLD_TRANSFORM = np.array([
    [ 6.1232340000e-17,  7.0710678100e-01, -7.0710678100e-01,  9.0000000000e-01],
    [ 9.9939082702e-01, -2.4677670772e-02, -2.4677670772e-02, -1.1500000000e-01],
    [-3.4899496703e-02, -7.0667603065e-01, -7.0667603065e-01,  7.3000000000e-01],
    [ 0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00],
])

# 相机内参（L515 RGB相机，1280x720）
# 使用数据采集时RealSense自动获取的典型内参
CAMERA_INTRINSICS = {
    'width': 1280,
    'height': 720,
    'fx': 915.0,      # RGB相机的焦距（基于70° FOV计算）
    'fy': 915.0,      # RGB相机的焦距
    'cx': 640.0,      # 图像中心X
    'cy': 360.0       # 图像中心Y
}

# 深度缩放因子（L515: 0.00025米/单位）
DEPTH_SCALE = 0.00025

# T_link2viz: RealSense相机坐标系到可视化坐标系的转换
# 这个转换用于点云处理，匹配数据采集时的坐标系
T_link2viz = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

# 工作空间边界（世界坐标系，用于裁剪点云）
# 与数据采集脚本保持一致
WORK_SPACE = [
    [0.3, 0.7],      # X范围（机器人前方）
    [-0.3, 0.3],    
    [0.285, 0.8]      # Z范围（高度，桌子表面以上）
]

# 点云处理参数
RESCALE_FACTOR = DEPTH_SCALE
CAMERA_FOVY = 45  # L515垂直FOV（度）

print("="*60)
print("真实环境相机参数 (RealSense L515)")
print("="*60)
print(f"相机类型: RealSense L515")
print(f"RGB分辨率: {CAMERA_INTRINSICS['width']}x{CAMERA_INTRINSICS['height']}")
print(f"深度分辨率: 640x480")
print(f"深度缩放因子: {DEPTH_SCALE}")
print(f"相机位置 (米): [{CAMERA_TO_WORLD_TRANSFORM[0,3]:.3f}, {CAMERA_TO_WORLD_TRANSFORM[1,3]:.3f}, {CAMERA_TO_WORLD_TRANSFORM[2,3]:.3f}]")
print(f"工作空间: {WORK_SPACE}")
print("="*60)
###################################################################################################


def get_camera_transform():
    """
    获取相机变换矩阵
    返回用于点云处理的相机参数
    """
    return {
        'T_link2viz': T_link2viz,
        'T_camera_to_world': CAMERA_TO_WORLD_TRANSFORM,
        'camera_intrinsics': CAMERA_INTRINSICS,
        'depth_scale': DEPTH_SCALE,
        'workspace': WORK_SPACE
    }


def get_camera_intrinsics():
    """
    获取相机内参
    返回CameraInfo格式的内参
    """
    from realsense_camera import CameraInfo
    return CameraInfo(
        width=CAMERA_INTRINSICS['width'],
        height=CAMERA_INTRINSICS['height'],
        fx=CAMERA_INTRINSICS['fx'],
        fy=CAMERA_INTRINSICS['fy'],
        cx=CAMERA_INTRINSICS['cx'],
        cy=CAMERA_INTRINSICS['cy'],
        scale=DEPTH_SCALE
    )


if __name__ == "__main__":
    # 测试相机变换
    transform = get_camera_transform()

    print("\n相机变换矩阵:")
    print("T_link2viz:")
    print(transform['T_link2viz'])

    print("\nT_camera_to_world:")
    print(transform['T_camera_to_world'])

    print("\n相机内参:")
    print(f"fx: {CAMERA_INTRINSICS['fx']}")
    print(f"fy: {CAMERA_INTRINSICS['fy']}")
    print(f"cx: {CAMERA_INTRINSICS['cx']}")
    print(f"cy: {CAMERA_INTRINSICS['cy']}")
