"""
仿真环境相机参数配置
适配MuJoCo仿真中的demogen_camera
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

################################# Camera Calibration ##############################################
# MuJoCo仿真中的相机配置 (来自 grasping_demogen.xml)
# <camera mode="fixed" name="demogen_camera" pos="1.0 0 0.7" quat="0.56 0.43 0.43 0.56" fovy="45"/>

# 机器人坐标系到相机坐标系的变换
# 在仿真中，相机是固定的，不需要像真实相机那样进行标定
# 但是为了保持与DemoGen的兼容性，我们需要提供等效的参数

# 相机位置（世界坐标系）
# pos="1.0 0 0.7" 表示相机在机器人基座前方1.0m，高度0.7m
# 机器人基座通常在原点 (0, 0, 0)
CAMERA_POS = np.array([1.0, 0.0, 0.7])

# 相机姿态（四元数 [w, x, y, z]）
# quat="0.56 0.43 0.43 0.56" 表示相机朝向机器人基座
CAMERA_QUAT = np.array([0.56, 0.43, 0.43, 0.56])  # [w, x, y, z]

# 将MuJoCo四元数转换为scipy格式 [x, y, z, w]
CAMERA_QUAT_SCIPY = np.array([0.43, 0.43, 0.56, 0.56])  # [x, y, z, w]

# 深度缩放因子（仿真中不需要缩放，深度直接是米）
MUJOCO_SCALE = 1.0

# 坐标转换矩阵：从相机坐标系到世界坐标系
# MuJoCo相机坐标系：X向右，Y向上，Z向前
# DemoGen期望的坐标系：需要根据实际采集时的坐标系确定

# 在真实机器人采集时，T_link2viz用于转换RealSense相机的坐标系
# 在仿真中，我们需要模拟这个转换

# T_link2viz: 从相机坐标系到可视化坐标系的转换
# 对于MuJoCo相机，这个矩阵可以根据相机姿态计算
cam_rot = R.from_quat(CAMERA_QUAT_SCIPY).as_matrix()

# 构建完整的4x4变换矩阵
T_camera_to_world = np.eye(4)
T_camera_to_world[:3, :3] = cam_rot
T_camera_to_world[:3, 3] = CAMERA_POS

# T_link2viz: 用于点云处理的坐标转换
# 这个矩阵需要匹配实际数据采集时使用的坐标系
# 在仿真中，我们可以使用单位矩阵，因为点云已经在世界坐标系中
T_link2viz = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

# 工作空间边界（用于裁剪点云）
# 根据仿真环境中的物体位置设置
WORK_SPACE = [
    [0.2, 0.8],   # X范围（机器人前方）
    [-0.3, 0.3],  # Y范围（左右）
    [0.42, 0.7]   # Z范围（高度，桌子表面0.423m）
]

# 点云处理参数（与真实环境保持一致）
RESCALE_FACTOR = 1.0  # 深度缩放因子

print("="*60)
print("仿真环境相机参数")
print("="*60)
print(f"CAMERA_POS: {CAMERA_POS}")
print(f"CAMERA_QUAT (MuJoCo): {CAMERA_QUAT}")
print(f"CAMERA_QUAT (scipy): {CAMERA_QUAT_SCIPY}")
print(f"MuJoCo fovy: 45 degrees")
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
        'camera_pos': CAMERA_POS,
        'camera_quat': CAMERA_QUAT_SCIPY,
        'scale': MUJOCO_SCALE,
        'workspace': WORK_SPACE
    }


if __name__ == "__main__":
    # 测试相机变换
    transform = get_camera_transform()

    print("\n相机变换矩阵:")
    print("T_link2viz:")
    print(transform['T_link2viz'])

    print("\nT_camera_to_world:")
    print(T_camera_to_world)
