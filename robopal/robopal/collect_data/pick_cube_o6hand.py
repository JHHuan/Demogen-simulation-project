"""
Panda + O6Hand灵巧手键盘遥操作 Demo - 盖盖子 (标准数据格式)
功能：
1. 键盘控制末端xyz位置和灵巧手6个关节
2. 采集RGB图像、深度图、点云用于训练
3. 每次按键自动采集一帧

数据格式（标准绝对位置格式）：
- point_cloud: (T, 1024, 6) - XYZ+RGB颜色
- image: (T, 3, 84, 84) - RGB CHW格式
- depth: (T, 84, 84) - 深度图
- agent_pos: (T, 12) - [x,y,z,qx,qy,qz,thumb_yaw,thumb_pitch,index,middle,ring,pinky]
- action: (T, 12) - [x,y,z,qx,qy,qz,thumb_yaw,thumb_pitch,index,middle,ring,pinky]

说明：
- agent_pos和action维度相同（都是12维）
- 前6维是末端位姿（位置+四元数去掉w）
- 后6维是灵巧手关节角度
- 灵巧手关节顺序：[thumb_yaw, thumb_pitch, index, middle, ring, pinky]

键盘控制说明：
- 方向键 ↑/↓/←/→    : 控制末端在 x/y 平面移动（自动采集）
- Ctrl + ↑/↓         : 控制 z 轴移动 (↑上升, ↓下降)（自动采集）
- 数字键 1-6         : 控制灵巧手各关节
  * 1: 大拇指外展/内收 (thumb_yaw)
  * 2: 大拇指屈曲 (thumb_pitch)
  * 3: 食指屈曲 (index)
  * 4: 中指屈曲 (middle)
  * 5: 无名指屈曲 (ring)
  * 6: 小指屈曲 (pinky)
  * Shift + 数字: 反向运动
- o 键               : 灵巧手全部张开
- c 键               : 灵巧手全部闭合
- g 键               : 四指同时闭合（快捷抓取，自动录制一帧）
- f 键               : 四指同时张开（快捷释放，自动录制一帧）
- 空格键             : 手动录制当前状态一帧
- r 键               : 重置环境
- a 键               : 保存轨迹为pickle文件
- x 键               : 清除当前轨迹
- ESC                : 退出

环境配置：
- 机器人: Panda + O6Hand灵巧手 (6 DOF)
- 控制器: CARTIK (笛卡尔空间逆运动学)
- 场景: grasping_demogen（盒子开盖任务）
- 相机: demogen_camera (机械臂正前方, pos=[1.0, 0, 0.7], fovy=45)

XML路径: /home/hjh/git_code/demogen/robopal/robopal/assets/scenes/grasping_demogen.xml
物体XML: /home/hjh/git_code/demogen/robopal/robopal/assets/objects/metaworld_box/metaworld_box.xml
"""

import numpy as np
import logging
import os
import pickle
import math
import mujoco
from datetime import datetime
from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaWithO6Hand
from robopal.devices.keyboard_o6hand import O6HandKeyboard
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


# ==========================================
# DemoGen轨迹保存函数
# ==========================================

def save_trajectory(point_cloud_list, image_list, depth_list, state_list, action_list, save_path):
    """
    保存轨迹为pickle格式（标准绝对位置格式）

    格式说明（12维）：
    - agent_pos: (T, 12) - [x,y,z,qx,qy,qz,thumb_yaw,thumb_pitch,index,middle,ring,pinky]
    - action: (T, 12) - [x,y,z,qx,qy,qz,thumb_yaw,thumb_pitch,index,middle,ring,pinky]
    """
    # 转换为numpy数组
    point_cloud_array = np.stack(point_cloud_list, axis=0)

    # 图像格式转换: (T, 84, 84, 3) → (T, 3, 84, 84) HWC → CHW
    image_array_hwc = np.stack(image_list, axis=0)
    image_array = np.transpose(image_array_hwc, (0, 3, 1, 2))

    depth_array = np.stack(depth_list, axis=0)
    state_array = np.stack(state_list, axis=0)
    action_array = np.stack(action_list, axis=0)

    # 组装数据
    data = {
        'point_cloud': point_cloud_array,  # (T, 1024, 6) XYZ+RGB
        'image': image_array,                  # (T, 3, 84, 84) CHW格式
        'depth': depth_array,                  # (T, 84, 84)
        'agent_pos': state_array,              # (T, 12) 当前位置
        'action': action_array                 # (T, 12) 目标位置
    }

    # 保存为pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    logging.info(f"轨迹已保存: {save_path}")
    logging.info(f"  轨迹长度: {len(point_cloud_list)} 帧")
    logging.info(f"  数据形状:")
    logging.info(f"    point_cloud: {point_cloud_array.shape}")
    logging.info(f"    image: {image_array.shape}")
    logging.info(f"    depth: {depth_array.shape}")
    logging.info(f"    agent_pos: {state_array.shape}")
    logging.info(f"    action: {action_array.shape}")

    return save_path


# ==========================================
# 点云生成工具函数
# ==========================================

def depth_to_point_cloud(depth, rgb, camera_name, mj_model, mj_data):
    """
    将 MuJoCo 深度图转换为点云（xyz + rgb）
    """
    height, width = depth.shape

    # 获取相机 ID
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # 计算相机内参
    fovy = math.radians(mj_model.cam_fovy[cam_id])
    f = height / (2 * math.tan(fovy / 2))
    cx = width / 2
    cy = height / 2

    # 创建像素坐标网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 深度图转换
    depth_meters = depth.copy()
    depth_meters = np.clip(depth_meters, 0.02, 2.0)

    # 计算 3D 点坐标（相机坐标系）
    z = depth_meters
    x = (u - cx) * z / f
    y = (v - cy) * z / f

    # 获取相机位姿
    cam_pos = mj_model.cam_pos[cam_id].copy()
    cam_mat = mj_model.cam_mat0[cam_id]
    cam_mat_3x3 = cam_mat.reshape(3, 3)

    # 坐标系转换
    rot_x_180 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    # 转换点云到世界坐标系
    points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points_cam_aligned = points_cam @ rot_x_180.T
    points_world = (cam_mat_3x3 @ points_cam_aligned.T).T + cam_pos

    # 提取 RGB 颜色
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    # 合并位置和颜色
    point_cloud = np.concatenate([points_world, colors], axis=1)

    # 过滤无效点并裁剪工作空间
    valid_mask = (
        np.isfinite(point_cloud[:, 0]) &
        np.isfinite(point_cloud[:, 1]) &
        np.isfinite(point_cloud[:, 2]) &
        (point_cloud[:, 2] > 0.0) &
        (point_cloud[:, 2] < 1.5) &
        (point_cloud[:, 0] > -0.5) &
        (point_cloud[:, 0] < 1.5) &
        (point_cloud[:, 1] > -1.0) &
        (point_cloud[:, 1] < 1.0)
    )
    point_cloud = point_cloud[valid_mask]

    return point_cloud


# ==========================================
# 点云后处理函数 (DemoGen兼容)
# ==========================================

def crop_point_cloud(point_cloud, workspace):
    """工作空间裁剪"""
    mask = (
        (point_cloud[:, 0] >= workspace[0][0]) &
        (point_cloud[:, 0] <= workspace[0][1]) &
        (point_cloud[:, 1] >= workspace[1][0]) &
        (point_cloud[:, 1] <= workspace[1][1]) &
        (point_cloud[:, 2] >= workspace[2][0]) &
        (point_cloud[:, 2] <= workspace[2][1])
    )
    return point_cloud[mask]


def cluster_dbscan(point_cloud, eps=0.03, min_samples=5, min_cluster_size=20):
    """DBSCAN聚类去噪"""
    from sklearn.cluster import DBSCAN

    if point_cloud.shape[0] < 100:
        return point_cloud

    n_random_drop = min(3000, point_cloud.shape[0] // 3)
    if point_cloud.shape[0] > n_random_drop:
        indices = np.random.choice(point_cloud.shape[0], n_random_drop, replace=False)
        point_cloud = point_cloud[indices]

    xyz = point_cloud[:, :3]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    outlier_labels = unique_labels[counts < min_cluster_size]
    if -1 not in outlier_labels:
        outlier_labels = np.append(outlier_labels, -1)

    mask = ~np.isin(labels, outlier_labels)

    if np.sum(mask) == 0:
        return point_cloud

    return point_cloud[mask]


def fps_sampling(point_cloud, n_points=1024):
    """FPS 采样"""
    xyz = point_cloud[:, :3]
    n_samples = xyz.shape[0]

    if n_samples <= n_points:
        indices = np.random.choice(n_samples, n_points, replace=True)
        return point_cloud[indices]

    indices = np.random.choice(n_samples, n_points, replace=False)
    return point_cloud[indices]


def preprocess_point_cloud(point_cloud, workspace, n_points=1024, debug=False):
    """完整的点云预处理流程 (DemoGen格式: XYZ+RGB)"""
    if debug:
        logging.info(f"  点云预处理开始: {point_cloud.shape[0]} 个点")

    point_cloud = crop_point_cloud(point_cloud, workspace)

    if point_cloud.shape[0] > 5000:
        point_cloud = cluster_dbscan(point_cloud)

    point_cloud = fps_sampling(point_cloud, n_points)
    # DemoGen格式：(1024, 6) [x, y, z, r, g, b] - 保留RGB颜色

    return point_cloud


def resize_image(image, target_size=(84, 84)):
    """Resize图像到目标尺寸"""
    from PIL import Image

    if len(image.shape) == 2:
        img_pil = Image.fromarray(image)
    else:
        img_pil = Image.fromarray(image)

    img_resized = img_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img_resized)


# ==========================================
# O6Hand灵巧手控制类
# ==========================================

class O6HandController:
    """O6Hand灵巧手控制器"""

    def __init__(self):
        # 6个关节的当前角度 - 初始状态为全部张开（0值）
        self.joint_angles = np.array([
            0.0,    # thumb_yaw (张开)
            0.0,    # thumb_pitch (张开)
            0.0,    # index (张开)
            0.0,    # middle (张开)
            0.0,    # ring (张开)
            0.0     # pinky (张开)
        ])

        # 关节范围（0=张开，最大值=闭合）
        self.joint_limits = [
            (0.0, 1.54),    # thumb_yaw
            (0.0, 0.52),    # thumb_pitch
            (0.0, 1.57),    # index
            (0.0, 1.57),    # middle
            (0.0, 1.57),    # ring
            (0.0, 1.57)     # pinky
        ]

        self.step_size = 0.1  # 每次按键的变化量

    def adjust_joint(self, joint_idx, direction=1):
        """
        调整指定关节

        Args:
            joint_idx: 关节索引 (0-5)
            direction: 1为增加，-1为减少
        """
        if 0 <= joint_idx < 6:
            self.joint_angles[joint_idx] += direction * self.step_size
            # 限制在范围内
            min_val, max_val = self.joint_limits[joint_idx]
            self.joint_angles[joint_idx] = np.clip(
                self.joint_angles[joint_idx], min_val, max_val
            )

    def open_all(self):
        """所有手指张开（0值）"""
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def close_all(self):
        """所有手指闭合（最大值）"""
        self.joint_angles = np.array([1.54, 0.52, 1.57, 1.57, 1.57, 1.57])

    def reset(self):
        """重置到张开状态"""
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def grasp_four_fingers(self):
        """四指同时闭合一半（保留大拇指张开）"""
        # thumb_yaw保持张开，其他手指闭合一半
        self.joint_angles = np.array([
            0.0,    # thumb_yaw (张开)
            0.0,    # thumb_pitch (张开)
            0.78,   # index (闭合一半，1.57/2)
            0.78,   # middle (闭合一半，1.57/2)
            0.78,   # ring (闭合一半，1.57/2)
            0.78    # pinky (闭合一半，1.57/2)
        ])

    def release_four_fingers(self):
        """四指同时张开（保留大拇指张开）"""
        # thumb_yaw保持张开，其他手指张开
        self.joint_angles = np.array([
            0.0,    # thumb_yaw (张开)
            0.0,    # thumb_pitch (张开)
            0.0,    # index (张开)
            0.0,    # middle (张开)
            0.0,    # ring (张开)
            0.0     # pinky (张开)
        ])

    def get_action(self):
        """获取当前灵巧手动作"""
        return self.joint_angles.copy()

# ==========================================

def keyboard_teleop_panda_o6hand_with_camera():
    """使用键盘遥操作 Panda + O6Hand灵巧手，支持图像采集"""

    # 创建环境
    env = RobotEnv(
        robot=PandaWithO6Hand,
        render_mode='human',
        is_render_camera_offscreen=True,
        camera_in_render='demogen_camera',
        control_freq=100,
        controller='CARTIK',
    )

    # 初始化键盘设备
    keyboard = O6HandKeyboard(pos_scale=0.01, rot_scale=0.01)
    keyboard.start()

    # 初始化灵巧手控制器
    hand_controller = O6HandController()

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 获取初始位姿
    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])

    # 灵巧手初始状态（半开）
    hand_action = hand_controller.get_action()
    env.robot.end['agent0'].apply_action(hand_action)
    env.step(action)

    # 目录设置
    save_dir = os.path.join(
        os.path.dirname(__file__),
        'collected_data_o6hand',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(save_dir, exist_ok=True)

    # 获取脚本名称
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # SAM mask保存目录
    sam_mask_dir = "/home/hjh/git_code/demogen/DemoGen-master/data/sam_mask_o6hand"
    os.makedirs(sam_mask_dir, exist_ok=True)

    # 轨迹数据
    point_cloud_traj = []
    image_traj = []
    depth_traj = []
    state_traj = []
    action_traj = []
    trajectory_count = 0

    # 第一次按键标志
    first_input = True

    logging.info("=" * 60)
    logging.info("Panda + O6Hand灵巧手键盘遥操作 Demo")
    logging.info("=" * 60)
    logging.info(f"数据保存目录: {save_dir}")
    logging.info(f"SAM mask目录: {sam_mask_dir}")
    logging.info("")
    logging.info("键盘控制说明:")
    logging.info("  方向键 ↑/↓/←/→  : 控制末端在 x/y 平面移动")
    logging.info("  Ctrl + ↑/↓       : 控制 z 轴移动 (↑上升, ↓下降)")
    logging.info("  数字键 1-6       : 控制灵巧手各关节")
    logging.info("    1: 大拇指外展/内收  2: 大拇指屈曲")
    logging.info("    3: 食指             4: 中指")
    logging.info("    5: 无名指           6: 小指")
    logging.info("  Shift + 数字      : 反向运动")
    logging.info("  o 键             : 灵巧手全部张开")
    logging.info("  c 键             : 灵巧手全部闭合")
    logging.info("  空格键           : 手动录制一帧")
    logging.info("  r 键             : 重置环境")
    logging.info("  a 键             : 保存轨迹")
    logging.info("  ESC              : 退出")
    logging.info("=" * 60)

    frame_count = 0
    last_action = action.copy()
    last_hand_action = hand_action.copy()
    last_had_input = False

    # =========================================================
    # 初始化离屏渲染器
    # =========================================================
    logging.info("初始化离屏渲染器...")
    high_res_renderer = mujoco.Renderer(env.mj_model, height=1080, width=1920)
    rgb_renderer = mujoco.Renderer(env.mj_model)
    depth_renderer = mujoco.Renderer(env.mj_model)
    depth_renderer.enable_depth_rendering()
    # =========================================================

    # =========================================================
    # 保存初始场景的高分辨率RGB图像
    # =========================================================
    logging.info("等待环境稳定...")
    for _ in range(100):
        env.step(action)
        env.robot.end['agent0'].apply_action(hand_action)

    logging.info("采集初始场景高分辨率图像...")
    high_res_renderer.update_scene(env.mj_data, camera='demogen_camera')
    high_res_image = high_res_renderer.render()
    high_res_image = high_res_image[:, :, ::-1]

    source_image_path = os.path.join(sam_mask_dir, "source.jpg")
    from PIL import Image
    img_pil = Image.fromarray(high_res_image.astype('uint8'))
    img_pil.save(source_image_path, quality=95)
    logging.info(f"✓ 初始场景图像已保存: {source_image_path}")
    logging.info(f"  形状: {high_res_image.shape}")
    # =========================================================

    try:
        while not keyboard._exit_flag:
            pos_offset, rot_offset = keyboard.get_outputs()

            # 检测是否有输入
            has_input_now = (
                np.any(np.abs(pos_offset) > 1e-6) or
                np.any(rot_offset != np.eye(3))
            )

            has_input = has_input_now and not last_had_input
            last_had_input = has_input_now

            # 更新末端位姿
            action[:3] += pos_offset
            action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(rot_offset))

            # 限制工作空间
            action[0] = np.clip(action[0], 0.1, 0.8)
            action[1] = np.clip(action[1], -0.35, 0.35)
            action[2] = np.clip(action[2], 0.0, 0.8)

            # 更新灵巧手动作
            joint_controls, open_all, close_all, grasp_four, release_four = keyboard.get_hand_joint_control()

            # 处理灵巧手控制
            if open_all:
                hand_controller.open_all()
                logging.info(f"[灵巧手] 全部张开")
            elif close_all:
                hand_controller.close_all()
                logging.info(f"[灵巧手] 全部闭合")
            elif grasp_four:
                hand_controller.grasp_four_fingers()
                logging.info(f"[灵巧手] 四指闭合（快捷抓取）")
                # 四指闭合后等待稳定并自动录制一帧
                logging.info(f"[灵巧手] 等待手指闭合稳定...")
                for _ in range(50):  # 等待50步让手指完全闭合
                    hand_action = hand_controller.get_action()
                    env.robot.end['agent0'].apply_action(hand_action)
                    env.step(action)
                logging.info(f"[灵巧手] 录制抓取帧")
                # 手动触发录制
                keyboard._record_frame_flag = True
            elif release_four:
                hand_controller.release_four_fingers()
                logging.info(f"[灵巧手] 四指张开（释放）")
                # 四指张开后等待稳定并自动录制一帧
                logging.info(f"[灵巧手] 等待手指张开稳定...")
                for _ in range(50):  # 等待50步让手指完全张开
                    hand_action = hand_controller.get_action()
                    env.robot.end['agent0'].apply_action(hand_action)
                    env.step(action)
                logging.info(f"[灵巧手] 录制释放帧")
                # 手动触发录制
                keyboard._record_frame_flag = True
            elif len(joint_controls) > 0:
                # 处理单个关节控制
                for joint_idx, direction in joint_controls:
                    old_angle = hand_controller.joint_angles[joint_idx]
                    hand_controller.adjust_joint(joint_idx, direction)
                    new_angle = hand_controller.joint_angles[joint_idx]
                    joint_names = ['thumb_yaw', 'thumb_pitch', 'index', 'middle', 'ring', 'pinky']
                    logging.info(f"[灵巧手] {joint_names[joint_idx]}: {old_angle:.3f} -> {new_angle:.3f}")

            # 每一步都应用灵巧手动作（持续控制）
            hand_action = hand_controller.get_action()
            env.robot.end['agent0'].apply_action(hand_action)

            env.step(action)

            if hasattr(env.renderer, 'exit_flag'):
                env.renderer.exit_flag = False

            # ========== 每次按键自动采集一帧 ==========
            if has_input:
                # 第一次按键采集高分辨率图像
                if first_input:
                    logging.info(f"\n[SAM Mask] 第一次按键，采集高分辨率图像...")
                    high_res_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    high_res_image = high_res_renderer.render()
                    high_res_image = high_res_image[:, :, ::-1]

                    sam_mask_path = os.path.join(sam_mask_dir, "source.jpg")
                    from PIL import Image
                    img_pil = Image.fromarray(high_res_image.astype('uint8'))
                    img_pil.save(sam_mask_path, quality=95)
                    logging.info(f"✓ 高分辨率图像已保存: {sam_mask_path}")

                    first_input = False

                frame_count += 1
                logging.info(f"\n添加帧 #{frame_count} 到轨迹...")

                try:
                    # 渲染深度图和RGB图像
                    depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    depth_image = depth_renderer.render()

                    rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    rgb_image = rgb_renderer.render()
                    rgb_image = rgb_image[:, :, ::-1]

                    # 图像resize到84x84
                    rgb_resized = resize_image(rgb_image, (84, 84))
                    depth_resized = resize_image(depth_image, (84, 84))

                    # 生成点云
                    point_cloud = depth_to_point_cloud(
                        depth_image,
                        rgb_image,
                        'demogen_camera',
                        env.mj_model,
                        env.mj_data
                    )

                    # 点云后处理
                    workspace = [
                        [0.12, 0.8],   # X范围
                        [-0.5, 0.5],  # Y范围
                        [0.426, 0.7]  # Z范围
                    ]
                    point_cloud_processed = preprocess_point_cloud(
                        point_cloud,
                        workspace=workspace,
                        n_points=1024,
                        debug=False
                    )

                    # 计算状态和action（都是12D）
                    end_effector_pos = action[:3]
                    end_effector_quat = action[3:]

                    # agent_pos: 当前位置（12维）
                    agent_pos_12d = np.concatenate([
                        end_effector_pos,          # [x, y, z]
                        end_effector_quat[1:],     # [qx, qy, qz]
                        hand_action                # [6个灵巧手关节]
                    ])

                    # action: 目标位置（12维）
                    action_12d = np.concatenate([
                        action[:3],              # [x, y, z]
                        action[3:][1:],          # [qx, qy, qz]
                        hand_action              # [6个灵巧手关节]
                    ])

                    # 添加到列表
                    point_cloud_traj.append(point_cloud_processed)
                    image_traj.append(rgb_resized)
                    depth_traj.append(depth_resized)
                    state_traj.append(agent_pos_12d)
                    action_traj.append(action_12d)

                    logging.info(f"  帧 #{frame_count} 已添加 (当前轨迹共 {len(point_cloud_traj)} 帧)")
                    logging.info(f"    末端位置: [{end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f}, {end_effector_pos[2]:.3f}]")
                    logging.info(f"    灵巧手: {np.round(hand_action, 2)}")

                except Exception as e:
                    import traceback
                    logging.error(f"  添加帧失败: {e}")
                    logging.error(f"  详细错误:\n{traceback.format_exc()}")

            last_action = action.copy()
            last_hand_action = hand_action.copy()

            # ========== 处理键盘命令 ==========
            # 保存轨迹
            if hasattr(keyboard, '_save_trajectory_flag') and keyboard._save_trajectory_flag:
                if len(point_cloud_traj) == 0:
                    logging.warning(f"\n当前轨迹为空。")
                else:
                    trajectory_count += 1
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = os.path.join(save_dir, f'{script_name}_{timestamp:04d}_{trajectory_count:03d}.pkl')

                    logging.info(f"\n保存轨迹 #{trajectory_count}...")
                    try:
                        save_trajectory(
                            point_cloud_traj, image_traj, depth_traj,
                            state_traj, action_traj, save_path
                        )
                        logging.info(f"  轨迹保存成功！")
                    except Exception as e:
                        logging.error(f"  保存轨迹失败: {e}")

                keyboard._save_trajectory_flag = False

            # 清除轨迹
            if hasattr(keyboard, '_clear_trajectory_flag') and keyboard._clear_trajectory_flag:
                if len(point_cloud_traj) > 0:
                    logging.info(f"\n清除轨迹数据...")
                    point_cloud_traj.clear()
                    image_traj.clear()
                    depth_traj.clear()
                    state_traj.clear()
                    action_traj.clear()
                    frame_count = 0
                keyboard._clear_trajectory_flag = False

            # 手动录制一帧（空格键）
            if keyboard._record_frame_flag:
                frame_count += 1
                logging.info(f"\n[手动录制] 添加帧 #{frame_count}...")

                try:
                    depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    depth_image = depth_renderer.render()

                    rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    rgb_image = rgb_renderer.render()
                    rgb_image = rgb_image[:, :, ::-1]

                    rgb_resized = resize_image(rgb_image, (84, 84))
                    depth_resized = resize_image(depth_image, (84, 84))

                    point_cloud = depth_to_point_cloud(
                        depth_image, rgb_image, 'demogen_camera',
                        env.mj_model, env.mj_data
                    )

                    workspace = [[0.12, 0.8], [-0.5, 0.5], [0.426, 0.7]]
                    point_cloud_processed = preprocess_point_cloud(
                        point_cloud, workspace=workspace, n_points=1024
                    )

                    end_effector_pos = action[:3]
                    end_effector_quat = action[3:]

                    agent_pos_12d = np.concatenate([
                        end_effector_pos, end_effector_quat[1:], hand_action
                    ])
                    action_12d = np.concatenate([
                        action[:3], action[3:][1:], hand_action
                    ])

                    point_cloud_traj.append(point_cloud_processed)
                    image_traj.append(rgb_resized)
                    depth_traj.append(depth_resized)
                    state_traj.append(agent_pos_12d)
                    action_traj.append(action_12d)

                    logging.info(f"  ✓ 帧 #{frame_count} 已添加")

                except Exception as e:
                    import traceback
                    logging.error(f"  录制失败: {e}\n{traceback.format_exc()}")

                keyboard._record_frame_flag = False

            # 重置环境
            if keyboard._reset_flag:
                env.reset()
                init_pos = env.robot.get_end_xpos()
                init_quat = env.robot.get_end_xquat()
                action = np.concatenate([init_pos, init_quat])
                hand_controller.reset()
                keyboard._reset_flag = False
                logging.info("环境已重置")

    except KeyboardInterrupt:
        pass
    finally:
        # 紧急保存
        if 'point_cloud_traj' in locals() and len(point_cloud_traj) > 0:
            logging.warning("检测到程序退出，执行紧急保存...")
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'{script_name}_{timestamp}_autosave.pkl')

                save_trajectory(
                    point_cloud_traj, image_traj, depth_traj,
                    state_traj, action_traj, save_path
                )
                logging.info(f"✅ 紧急保存成功！文件位于: {save_path}")
            except Exception as e:
                logging.error(f"❌ 紧急保存失败: {e}")

        # 清理渲染器
        if 'rgb_renderer' in locals():
            rgb_renderer.close()
        if 'depth_renderer' in locals():
            depth_renderer.close()
        if 'high_res_renderer' in locals():
            high_res_renderer.close()
        env.close()
        logging.info("仿真结束")


# ==========================================
# 主函数
# ==========================================

if __name__ == "__main__":
    keyboard_teleop_panda_o6hand_with_camera()
