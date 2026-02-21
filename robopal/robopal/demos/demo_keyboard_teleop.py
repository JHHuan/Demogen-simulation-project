"""
Panda 机械臂键盘遥操作 Demo (标准数据格式)
功能：
1. 键盘控制末端xyz位置和爪子开闭
2. 采集RGB图像、深度图、点云用于训练
3. 每次按键自动采集一帧

数据格式（标准绝对位置格式）：
- point_cloud: (T, 1024, 6) - XYZ+RGB颜色
- image: (T, 3, 84, 84) - RGB CHW格式
- depth: (T, 84, 84) - 深度图
- agent_pos: (T, 7) - [x,y,z,qx,qy,qz,gripper]  (当前实际位置，绝对坐标)
- action: (T, 7) - [x,y,z,qx,qy,qz,gripper]     (目标位置，绝对坐标)

说明：
- agent_pos和action维度相同（都是7维）
- 都是绝对位置（世界坐标系）
- 使用四元数表示姿态
- gripper: 0=闭合, 0.04=张开

键盘控制说明：
- 方向键 ↑/↓/←/→    : 控制末端在 x/y 平面移动（自动采集）
- Ctrl + ↑/↓         : 控制 z 轴移动 (↑上升, ↓下降)（自动采集）
- CapsLock           : 切换爪子开/闭状态（自动采集）
- 空格键             : 手动录制当前状态一帧
- r 键               : 重置环境
- q 键               : 保存轨迹为pickle文件
- c 键               : 清除当前轨迹
- ESC                : 退出

环境配置：
- 机器人: Panda + PandaHand 爪子
- 控制器: CARTIK (笛卡尔空间逆运动学)
- 场景: 桌子上有一个绿色立方体
- 相机: demogen_camera (机械臂正前方, pos=[1.0, 0, 0.7], fovy=45)

XML路径: /home/hjh/git_code/demogen/robopal/robopal/assets/scenes/grasping_demogen.xml
"""

import numpy as np
import logging
import os
import pickle
import math
import mujoco
from datetime import datetime
from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaDemoGen
from robopal.devices import Keyboard
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


# ==========================================
# DemoGen轨迹保存函数
# ==========================================

def save_trajectory(point_cloud_list, image_list, depth_list, state_list, action_list, save_path):
    """
    保存轨迹为pickle格式（标准绝对位置格式）

    格式说明：
    - agent_pos: (T, 7) - [x,y,z,qx,qy,qz,gripper] (当前实际位置)
    - action: (T, 7) - [x,y,z,qx,qy,qz,gripper] (目标位置)
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
        'agent_pos': state_array,              # (T, 7) 当前位置
        'action': action_array                 # (T, 7) 目标位置
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


def save_point_cloud_as_ply(point_cloud, filepath):
    """保存点云为 PLY 格式"""
    n_points = point_cloud.shape[0]

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n_points):
            x, y, z = point_cloud[i, :3]
            r, g, b = point_cloud[i, 3:] * 255
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def save_point_cloud_as_npy(point_cloud, filepath):
    np.save(filepath, point_cloud)


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


def keyboard_teleop_panda_with_camera():
    """使用键盘遥操作 Panda 机械臂，支持图像采集"""

    # 创建环境
    env = RobotEnv(
        robot=PandaDemoGen,
        render_mode='human',                   
        is_render_camera_offscreen=True,      
        camera_in_render='demogen_camera',     
        control_freq=100,
        controller='CARTIK',
    )

    if hasattr(env.renderer, 'enable_viewer_keyboard'):
        env.renderer.enable_viewer_keyboard = False

    # 初始化键盘设备
    keyboard = Keyboard(pos_scale=0.01, rot_scale=0.01)
    keyboard.start()

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 获取初始位姿
    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])

    # 爪子初始状态
    keyboard._gripper_flag = 0
    env.robot.end['agent0'].open()
    env.step(action)

    # 目录设置
    save_dir = os.path.join(
        os.path.dirname(__file__),
        'collected_data',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(save_dir, exist_ok=True)

    # SAM mask保存目录
    sam_mask_dir = "/home/hjh/git_code/demogen/DemoGen-master/data/sam_mask/0216-cube"
    os.makedirs(sam_mask_dir, exist_ok=True)

    # 轨迹数据
    point_cloud_traj = []
    image_traj = []
    depth_traj = []
    state_traj = []
    action_traj = []
    trajectory_count = 0

    # 第一次按键标志（用于采集高分辨率图像）
    first_input = True

    logging.info("Panda 机械臂键盘遥操作 Demo (Ready)")
    logging.info(f"数据保存目录: {save_dir}")
    logging.info(f"SAM mask目录: {sam_mask_dir}")

    frame_count = 0
    last_action = action.copy()
    last_gripper_flag = keyboard._gripper_flag
    last_had_input = False

    # 夹爪状态变化标志（用于在夹爪动作完成后额外录制一帧）
    gripper_state_changed = False

    # =========================================================
    # 初始化离屏渲染器
    # =========================================================
    logging.info("初始化离屏渲染器...")
    # 高分辨率 RGB 渲染器 (1920x1080, 用于SAM mask)
    high_res_renderer = mujoco.Renderer(env.mj_model, height=1080, width=1920)
    # 低分辨率 RGB 渲染器 (默认, 用于策略训练)
    rgb_renderer = mujoco.Renderer(env.mj_model)
    # 深度渲染器
    depth_renderer = mujoco.Renderer(env.mj_model)
    depth_renderer.enable_depth_rendering()
    # =========================================================

    # =========================================================
    # 保存初始场景的高分辨率RGB图像
    # =========================================================
    logging.info("等待环境稳定...")
    # 让环境稳定，确保物体落下
    for _ in range(100):
        env.step(action)

    # 渲染高分辨率RGB图像
    logging.info("采集初始场景高分辨率图像...")
    high_res_renderer.update_scene(env.mj_data, camera='demogen_camera')
    high_res_image = high_res_renderer.render()
    high_res_image = high_res_image[:, :, ::-1]  # RGB -> BGR (用于OpenCV/PIL保存)

    # 保存到指定路径
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

            has_input_now = (
                np.any(np.abs(pos_offset) > 1e-6) or
                np.any(rot_offset != np.eye(3)) or
                keyboard._gripper_flag != last_gripper_flag
            )

            # 检测夹爪状态变化（用于额外录制一帧）
            if keyboard._gripper_flag != last_gripper_flag:
                gripper_state_changed = True

            has_input = has_input_now and not last_had_input
            last_had_input = has_input_now

            action[:3] += pos_offset
            action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(rot_offset))

            action[0] = np.clip(action[0], 0.3, 0.6)
            action[1] = np.clip(action[1], -0.2, 0.2)
            action[2] = np.clip(action[2], 0.0, 0.8)

            if keyboard._gripper_flag:
                env.robot.end['agent0'].close()
            else:
                env.robot.end['agent0'].open()

            # [FIX] 不需要暂停渲染了，因为我们使用独立的 renderer
            env.step(action)

            if hasattr(env.renderer, 'exit_flag'):
                env.renderer.exit_flag = False

            last_gripper_flag = keyboard._gripper_flag

            # ========== DemoGen方式：每次按键自动采集一帧 ==========
            if has_input:
                # 【第一次按键】采集并保存高分辨率图像（机械臂还没动）
                if first_input:
                    logging.info(f"\n[SAM Mask] 第一次按键，采集高分辨率图像...")
                    high_res_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    high_res_image = high_res_renderer.render()
                    high_res_image = high_res_image[:, :, ::-1]  # RGB -> BGR

                    # 保存为source.jpg
                    sam_mask_path = os.path.join(sam_mask_dir, "source.jpg")
                    from PIL import Image
                    img_pil = Image.fromarray(high_res_image.astype('uint8'))
                    img_pil.save(sam_mask_path, quality=95)
                    logging.info(f"✓ 高分辨率图像已保存: {sam_mask_path}")
                    logging.info(f"  形状: {high_res_image.shape}")

                    first_input = False

                frame_count += 1
                logging.info(f"\n添加帧 #{frame_count} 到轨迹...")

                try:
                    # 渲染深度图和RGB图像
                    depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    depth_image = depth_renderer.render()

                    rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    rgb_image = rgb_renderer.render()
                    rgb_image = rgb_image[:, :, ::-1]  # RGB -> BGR

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
                        [0.1, 0.8],   # X范围
                        [-0.5, 0.5],  # Y范围
                        [0.423, 0.7]  # Z范围
                    ]
                    point_cloud_processed = preprocess_point_cloud(
                        point_cloud,
                        workspace=workspace,
                        n_points=1024,
                        debug=False
                    )

                    # 计算状态和action（都是7D: x,y,z,qx,qy,qz,gripper）
                    end_effector_pos = action[:3]
                    end_effector_quat = action[3:]

                    # gripper目标值（根据keyboard._gripper_flag确定）
                    # True=闭合(0.0), False=打开(0.04)
                    gripper_target = 0.0 if keyboard._gripper_flag else 0.04

                    # agent_pos: 当前实际位置（7维）
                    agent_pos_7d = np.concatenate([
                        end_effector_pos,     # [x, y, z]
                        end_effector_quat[1:],  # [qx, qy, qz] (注意：scipy格式是[x,y,z,w]，mujoco是[x,y,z,w])
                        [gripper_target]      # [gripper] 目标状态（0.0=闭合, 0.04=打开）
                    ])

                    # action: 目标位置（7维，与agent_pos同格式）
                    action_7d = np.concatenate([
                        action[:3],          # [x, y, z] 目标位置
                        action[3:][1:],      # [qx, qy, qz] 目标四元数
                        [gripper_target]     # [gripper] 目标状态（0.0=闭合, 0.04=打开）
                    ])

                    # 添加到列表
                    point_cloud_traj.append(point_cloud_processed)
                    image_traj.append(rgb_resized)
                    depth_traj.append(depth_resized)
                    state_traj.append(agent_pos_7d)
                    action_traj.append(action_7d)

                    logging.info(f"  帧 #{frame_count} 已添加 (当前轨迹共 {len(point_cloud_traj)} 帧)")

                    # 【夹爪状态变化时额外录制一帧】
                    # 如果夹爪状态发生了变化（从开到关或从关到开），等待几步让夹爪完全动作后再录制一帧
                    if gripper_state_changed:
                        logging.info(f"  检测到夹爪状态变化，等待夹爪完全动作...")
                        # 等待20步让夹爪完全闭合/打开
                        for wait_step in range(20):
                            env.step(action)
                            # 持续控制夹爪保持目标状态
                            if keyboard._gripper_flag:
                                env.robot.end['agent0'].close()
                            else:
                                env.robot.end['agent0'].open()

                        # 额外录制一帧
                        frame_count += 1
                        logging.info(f"  录制夹爪动作后的额外帧 #{frame_count}...")

                        try:
                            # 渲染深度图和RGB图像
                            depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
                            depth_image = depth_renderer.render()

                            rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
                            rgb_image = rgb_renderer.render()
                            rgb_image = rgb_image[:, :, ::-1]  # RGB -> BGR

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
                                [0.1, 0.8],   # X范围
                                [-0.5, 0.5],  # Y范围
                                [0.423, 0.7]  # Z范围
                            ]
                            point_cloud_processed = preprocess_point_cloud(
                                point_cloud,
                                workspace=workspace,
                                n_points=1024,
                                debug=False
                            )

                            # 计算状态和action（都是7D: x,y,z,qx,qy,qz,gripper）
                            end_effector_pos = action[:3]
                            end_effector_quat = action[3:]

                            # gripper目标值
                            gripper_target = 0.0 if keyboard._gripper_flag else 0.04

                            # agent_pos: 当前实际位置（7维）
                            agent_pos_7d = np.concatenate([
                                end_effector_pos,     # [x, y, z]
                                end_effector_quat[1:],  # [qx, qy, qz]
                                [gripper_target]      # [gripper]
                            ])

                            # action: 目标位置（7维）
                            action_7d = np.concatenate([
                                action[:3],          # [x, y, z]
                                action[3:][1:],      # [qx, qy, qz]
                                [gripper_target]     # [gripper]
                            ])

                            # 添加到列表
                            point_cloud_traj.append(point_cloud_processed)
                            image_traj.append(rgb_resized)
                            depth_traj.append(depth_resized)
                            state_traj.append(agent_pos_7d)
                            action_traj.append(action_7d)

                            logging.info(f"  ✓ 夹爪动作后帧 #{frame_count} 已添加 (当前轨迹共 {len(point_cloud_traj)} 帧)")

                        except Exception as e:
                            import traceback
                            logging.error(f"  添加夹爪动作后帧失败: {e}")
                            logging.error(f"  详细错误:\n{traceback.format_exc()}")

                        # 重置标志
                        gripper_state_changed = False

                except Exception as e:
                    import traceback
                    logging.error(f"  添加帧失败: {e}")
                    logging.error(f"  详细错误:\n{traceback.format_exc()}")

            last_action = action.copy()
            last_gripper_flag = keyboard._gripper_flag

            # ========== 保存轨迹 ==========
            if hasattr(keyboard, '_save_trajectory_flag') and keyboard._save_trajectory_flag:
                if len(point_cloud_traj) == 0:
                    logging.warning(f"\n当前轨迹为空。")
                else:
                    trajectory_count += 1
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = os.path.join(save_dir, f'trajectory_{timestamp:04d}_{trajectory_count:03d}.pkl')

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

            # ========== 清除轨迹 ==========
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

            # ========== 手动录制一帧（空格键）==========
            if keyboard._record_frame_flag:
                logging.info(f"\n[主循环] _record_frame_flag = {keyboard._record_frame_flag}")
                logging.info(f"[手动录制] 检测到空格键按下！")
                frame_count += 1
                logging.info(f"[手动录制] 添加帧 #{frame_count} 到轨迹...")

                try:
                    # 渲染深度图和RGB图像
                    depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    depth_image = depth_renderer.render()

                    rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
                    rgb_image = rgb_renderer.render()
                    rgb_image = rgb_image[:, :, ::-1]  # RGB -> BGR

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
                        [0.1, 0.8],   # X范围
                        [-0.5, 0.5],  # Y范围
                        [0.423, 0.7]  # Z范围
                    ]
                    point_cloud_processed = preprocess_point_cloud(
                        point_cloud,
                        workspace=workspace,
                        n_points=1024,
                        debug=False
                    )

                    # 计算状态和action（都是7D: x,y,z,qx,qy,qz,gripper）
                    end_effector_pos = action[:3]
                    end_effector_quat = action[3:]

                    # gripper目标值
                    gripper_target = 0.0 if keyboard._gripper_flag else 0.04

                    # agent_pos: 当前实际位置（7维）
                    agent_pos_7d = np.concatenate([
                        end_effector_pos,     # [x, y, z]
                        end_effector_quat[1:],  # [qx, qy, qz]
                        [gripper_target]      # [gripper]
                    ])

                    # action: 目标位置（7维）
                    action_7d = np.concatenate([
                        action[:3],          # [x, y, z]
                        action[3:][1:],      # [qx, qy, qz]
                        [gripper_target]     # [gripper]
                    ])

                    # 添加到列表
                    point_cloud_traj.append(point_cloud_processed)
                    image_traj.append(rgb_resized)
                    depth_traj.append(depth_resized)
                    state_traj.append(agent_pos_7d)
                    action_traj.append(action_7d)

                    logging.info(f"  ✓ 手动录制帧 #{frame_count} 已添加 (当前轨迹共 {len(point_cloud_traj)} 帧)")

                except Exception as e:
                    import traceback
                    logging.error(f"  手动录制帧失败: {e}")
                    logging.error(f"  详细错误:\n{traceback.format_exc()}")

                keyboard._record_frame_flag = False

            # ========== 重置 ==========
            if keyboard._reset_flag:
                env.reset()
                init_pos = env.robot.get_end_xpos()
                init_quat = env.robot.get_end_xquat()
                action = np.concatenate([init_pos, init_quat])
                keyboard._gripper_flag = 0
                keyboard._reset_flag = False
                logging.info("环境已重置")

    except KeyboardInterrupt:
        pass
    finally:
        # ==========================================
        # [NEW] 退出前紧急保存机制 (防止数据丢失)
        # ==========================================
        if 'point_cloud_traj' in locals() and len(point_cloud_traj) > 0:
            logging.warning("检测到程序退出且有未保存的数据，正在执行紧急保存...")
            try:
                # 生成带 _autosave 后缀的文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'trajectory_{timestamp}_autosave.pkl')
                
                save_trajectory(
                    point_cloud_traj, image_traj, depth_traj,
                    state_traj, action_traj, save_path
                )
                logging.info(f"✅ 紧急保存成功！文件位于: {save_path}")
            except Exception as e:
                logging.error(f"❌ 紧急保存失败: {e}")
        else:
            logging.info("无未保存数据或列表为空。")

        # [FIX] 清理渲染器
        if 'rgb_renderer' in locals():
            rgb_renderer.close()
        if 'depth_renderer' in locals():
            depth_renderer.close()
        if 'high_res_renderer' in locals():
            high_res_renderer.close()
        env.close()
        logging.info("仿真结束")


def keyboard_teleop_panda_simple():
    """简化版：仅遥操作，不保存数据"""
    env = RobotEnv(
        robot=PandaDemoGen,
        render_mode='human',
        control_freq=100,
        controller='CARTIK',
    )
    keyboard = Keyboard(pos_scale=0.01, rot_scale=0.01)
    keyboard.start()
    env.reset()
    env.controller.reference = 'world'

    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])
    keyboard._gripper_flag = 0
    env.robot.end['agent0'].open()

    logging.info("Panda 机械臂键盘遥操作 Demo (简化版)")

    try:
        while not keyboard._exit_flag:
            pos_offset, rot_offset = keyboard.get_outputs()
            action[:3] += pos_offset
            action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(rot_offset))
            action[0] = np.clip(action[0], 0.3, 0.6)
            action[1] = np.clip(action[1], -0.2, 0.2)
            action[2] = np.clip(action[2], 0.0, 1)

            if keyboard._gripper_flag:
                env.robot.end['agent0'].close()
            else:
                env.robot.end['agent0'].open()

            env.step(action)

            if keyboard._reset_flag:
                env.reset()
                init_pos = env.robot.get_end_xpos()
                init_quat = env.robot.get_end_xquat()
                action = np.concatenate([init_pos, init_quat])
                keyboard._gripper_flag = 0
                keyboard._reset_flag = False

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        logging.info("仿真结束")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        keyboard_teleop_panda_simple()
    else:
        keyboard_teleop_panda_with_camera()