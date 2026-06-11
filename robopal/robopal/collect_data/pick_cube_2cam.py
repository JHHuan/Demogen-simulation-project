"""
Panda 机械臂键盘遥操作 - 同时录制两种点云数据

一次操作, 按 a 保存时输出两个文件:
  1. *_2cam.pkl : 双相机 (cam_left 512 + cam_right 512 = 1024 点)
  2. *_1cam.pkl : 单相机 (cam_front 1024 点)

场景包含 3 个相机:
  - cam_front: 正前方 (pos=[1.5, 0, 0.8], fovy=45)
  - cam_left:  前方偏左45°
  - cam_right: 前方偏右45°

image/depth 统一取自 cam_front

数据格式 (两个文件格式一致):
- point_cloud: (T, 1024, 6) - XYZ+RGB
- image: (T, 3, 84, 84) - cam_front RGB CHW
- depth: (T, 84, 84) - cam_front 深度
- agent_pos: (T, 7) - [x,y,z,qx,qy,qz,gripper]
- action: (T, 7) - [x,y,z,qx,qy,qz,gripper]
"""

import numpy as np
import logging
import os
import pickle
import math
import mujoco
from datetime import datetime
from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaSingleCube3cam
from robopal.devices import Keyboard
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)

CAM_FRONT = 'cam_front'
CAM_LEFT = 'cam_left'
CAM_RIGHT = 'cam_right'
N_POINTS = 1024
POINT_PER_DUAL_CAM = 512

WORKSPACE = [
    [0.12, 0.8],
    [-0.5, 0.5],
    [0.425, 0.7],
]


# ==========================================
# 点云工具
# ==========================================

def depth_to_point_cloud(depth, rgb, camera_name, mj_model, mj_data):
    height, width = depth.shape
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = math.radians(mj_model.cam_fovy[cam_id])
    f = height / (2 * math.tan(fovy / 2))
    cx, cy = width / 2, height / 2
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    depth_meters = np.clip(depth.copy(), 0.02, 2.0)
    z = depth_meters
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    cam_pos = mj_model.cam_pos[cam_id].copy()
    cam_mat = mj_model.cam_mat0[cam_id].reshape(3, 3)
    rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points_cam_aligned = points_cam @ rot_x_180.T
    points_world = (cam_mat @ points_cam_aligned.T).T + cam_pos
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    point_cloud = np.concatenate([points_world, colors], axis=1)
    valid_mask = (
        np.isfinite(point_cloud[:, 0]) & np.isfinite(point_cloud[:, 1]) &
        np.isfinite(point_cloud[:, 2]) &
        (point_cloud[:, 2] > 0.0) & (point_cloud[:, 2] < 1.5) &
        (point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] < 1.5) &
        (point_cloud[:, 1] > -1.0) & (point_cloud[:, 1] < 1.0)
    )
    return point_cloud[valid_mask]


def crop_point_cloud(pc, workspace=WORKSPACE):
    mask = (
        (pc[:, 0] >= workspace[0][0]) & (pc[:, 0] <= workspace[0][1]) &
        (pc[:, 1] >= workspace[1][0]) & (pc[:, 1] <= workspace[1][1]) &
        (pc[:, 2] >= workspace[2][0]) & (pc[:, 2] <= workspace[2][1])
    )
    return pc[mask]


def sample_points(pc, n):
    if len(pc) == 0:
        return np.zeros((n, pc.shape[1] if pc.ndim > 1 else 6), dtype=np.float32)
    if len(pc) >= n:
        idx = np.random.choice(len(pc), n, replace=False)
    else:
        idx = np.random.choice(len(pc), n, replace=True)
    return pc[idx].astype(np.float32)


# ==========================================
# 保存 (同时存两份)
# ==========================================

def save_both(pc_2cam_list, pc_1cam_list, img_list, depth_list,
              state_list, action_list, base_path):
    """同时保存 2cam 和 1cam 两个 pkl 文件"""
    img_arr = np.transpose(np.stack(img_list), (0, 3, 1, 2))

    # --- 2cam 文件 ---
    data_2cam = {
        'point_cloud': np.stack(pc_2cam_list),
        'image': img_arr,
        'depth': np.stack(depth_list),
        'agent_pos': np.stack(state_list),
        'action': np.stack(action_list),
    }
    path_2cam = base_path.replace('.pkl', '_2cam.pkl')
    with open(path_2cam, 'wb') as f:
        pickle.dump(data_2cam, f)

    # --- 1cam 文件 ---
    data_1cam = {
        'point_cloud': np.stack(pc_1cam_list),
        'image': img_arr,
        'depth': np.stack(depth_list),
        'agent_pos': np.stack(state_list),
        'action': np.stack(action_list),
    }
    path_1cam = base_path.replace('.pkl', '_1cam.pkl')
    with open(path_1cam, 'wb') as f:
        pickle.dump(data_1cam, f)

    logging.info(f"已保存 ({len(pc_2cam_list)} 帧):")
    logging.info(f"  2cam: {path_2cam}")
    logging.info(f"  1cam: {path_1cam}")


def resize_image(img, size=(84, 84)):
    from PIL import Image as PILImage
    return np.array(PILImage.fromarray(img).resize((size[1], size[0]), PILImage.BILINEAR))


# ==========================================
# 主循环
# ==========================================

def main():
    env = RobotEnv(
        robot=PandaSingleCube3cam,
        render_mode='human',
        is_render_camera_offscreen=True,
        camera_in_render=CAM_FRONT,
        control_freq=100,
        controller='CARTIK',
    )
    if hasattr(env.renderer, 'enable_viewer_keyboard'):
        env.renderer.enable_viewer_keyboard = False

    keyboard = Keyboard(pos_scale=0.025, rot_scale=0.01)
    keyboard.start()
    env.reset()
    env.controller.reference = 'world'

    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])
    keyboard._gripper_flag = 0
    env.robot.end['agent0'].open()
    env.step(action)

    save_dir = os.path.join(os.path.dirname(__file__), 'collected_data',
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # SAM mask保存目录
    sam_mask_dir = "/home/hjh/git_code/demogen/DemoGen-master/data/sam_mask"
    os.makedirs(sam_mask_dir, exist_ok=True)

    # 渲染器: 3 相机各一个 depth + rgb
    renderers = {}
    for cam in [CAM_FRONT, CAM_LEFT, CAM_RIGHT]:
        r_rgb = mujoco.Renderer(env.mj_model)
        r_depth = mujoco.Renderer(env.mj_model)
        r_depth.enable_depth_rendering()
        renderers[cam] = {'rgb': r_rgb, 'depth': r_depth}

    # 高分辨率 RGB 渲染器 (1920x1080, 用于SAM mask)
    high_res_renderer = mujoco.Renderer(env.mj_model, height=1080, width=1920)

    logging.info("等待环境稳定...")
    for _ in range(100):
        env.step(action)

    # 保存初始场景的高分辨率RGB图像
    logging.info("采集初始场景高分辨率图像...")
    high_res_renderer.update_scene(env.mj_data, camera=CAM_FRONT)
    high_res_image = high_res_renderer.render()
    high_res_image = high_res_image[:, :, ::-1]  # RGB -> BGR
    source_image_path = os.path.join(sam_mask_dir, "source.jpg")
    from PIL import Image
    img_pil = Image.fromarray(high_res_image.astype('uint8'))
    img_pil.save(source_image_path, quality=95)
    logging.info(f"✓ 初始场景图像已保存: {source_image_path}")
    logging.info(f"  形状: {high_res_image.shape}")

    # 两条轨迹分别存点云, 共享 image/depth/state/action
    pc_2cam_traj, pc_1cam_traj = [], []
    img_traj, depth_traj, state_traj, action_traj = [], [], [], []
    traj_count = 0
    frame_count = 0
    last_gripper = keyboard._gripper_flag
    last_had_input = False
    gripper_changed = False

    logging.info(f"三相机采集就绪 - 每次保存同时输出 _2cam.pkl 和 _1cam.pkl")

    try:
        while not keyboard._exit_flag:
            pos_offset, rot_offset = keyboard.get_outputs()
            has_now = (np.any(np.abs(pos_offset) > 1e-6) or
                       np.any(rot_offset != np.eye(3)) or
                       keyboard._gripper_flag != last_gripper)
            if keyboard._gripper_flag != last_gripper:
                gripper_changed = True
            has_input = has_now and not last_had_input
            last_had_input = has_now

            action[:3] += pos_offset
            action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(rot_offset))
            action[0] = np.clip(action[0], 0.2, 0.7)
            action[1] = np.clip(action[1], -0.3, 0.3)
            action[2] = np.clip(action[2], 0.0, 0.8)

            if keyboard._gripper_flag:
                env.robot.end['agent0'].close()
            else:
                env.robot.end['agent0'].open()
            env.step(action)
            if hasattr(env.renderer, 'exit_flag'):
                env.renderer.exit_flag = False
            last_gripper = keyboard._gripper_flag

            def capture_frame():
                """3 相机渲染 → 双路点云 + 共享 image/depth"""
                depths, rgbs = {}, {}
                for cam in [CAM_FRONT, CAM_LEFT, CAM_RIGHT]:
                    renderers[cam]['depth'].update_scene(env.mj_data, camera=cam)
                    depths[cam] = renderers[cam]['depth'].render()
                    renderers[cam]['rgb'].update_scene(env.mj_data, camera=cam)
                    rgbs[cam] = renderers[cam]['rgb'].render()[:, :, ::-1]

                # 双相机点云: left 512 + right 512
                pc_l = crop_point_cloud(depth_to_point_cloud(depths[CAM_LEFT], rgbs[CAM_LEFT], CAM_LEFT, env.mj_model, env.mj_data))
                pc_r = crop_point_cloud(depth_to_point_cloud(depths[CAM_RIGHT], rgbs[CAM_RIGHT], CAM_RIGHT, env.mj_model, env.mj_data))
                pc_2cam = np.concatenate([sample_points(pc_l, POINT_PER_DUAL_CAM),
                                          sample_points(pc_r, POINT_PER_DUAL_CAM)], axis=0)

                # 单相机点云: front 1024
                pc_f = crop_point_cloud(depth_to_point_cloud(depths[CAM_FRONT], rgbs[CAM_FRONT], CAM_FRONT, env.mj_model, env.mj_data))
                pc_1cam = sample_points(pc_f, N_POINTS)

                # image/depth 取 cam_front
                rgb_resized = resize_image(rgbs[CAM_FRONT], (84, 84))
                depth_resized = resize_image(depths[CAM_FRONT], (84, 84))

                gripper_target = 0.0 if keyboard._gripper_flag else 0.04
                agent_pos = np.concatenate([action[:3], action[3:][1:], [gripper_target]])
                action_7d = np.concatenate([action[:3], action[3:][1:], [gripper_target]])

                return pc_2cam, pc_1cam, rgb_resized, depth_resized, agent_pos, action_7d

            def add_frame(pc2, pc1, rgb, dep, st, act):
                pc_2cam_traj.append(pc2)
                pc_1cam_traj.append(pc1)
                img_traj.append(rgb)
                depth_traj.append(dep)
                state_traj.append(st)
                action_traj.append(act)

            if has_input:
                frame_count += 1
                try:
                    pc2, pc1, rgb, dep, st, act = capture_frame()
                    add_frame(pc2, pc1, rgb, dep, st, act)
                    logging.info(f"  帧 #{frame_count} ({len(pc_2cam_traj)} 帧)")
                    if gripper_changed:
                        for _ in range(20):
                            env.step(action)
                            if keyboard._gripper_flag:
                                env.robot.end['agent0'].close()
                            else:
                                env.robot.end['agent0'].open()
                        frame_count += 1
                        pc2, pc1, rgb, dep, st, act = capture_frame()
                        add_frame(pc2, pc1, rgb, dep, st, act)
                        logging.info(f"  夹爪帧 #{frame_count} ({len(pc_2cam_traj)} 帧)")
                        gripper_changed = False
                except Exception as e:
                    logging.error(f"  采集失败: {e}")

            if keyboard._record_frame_flag:
                frame_count += 1
                try:
                    pc2, pc1, rgb, dep, st, act = capture_frame()
                    add_frame(pc2, pc1, rgb, dep, st, act)
                    logging.info(f"  手动帧 #{frame_count} ({len(pc_2cam_traj)} 帧)")
                except Exception as e:
                    logging.error(f"  手动采集失败: {e}")
                keyboard._record_frame_flag = False

            # 保存 (a 键) → 同时输出两个文件
            if hasattr(keyboard, '_save_trajectory_flag') and keyboard._save_trajectory_flag:
                if pc_2cam_traj:
                    traj_count += 1
                    base = os.path.join(save_dir,
                        f'{script_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{traj_count:03d}.pkl')
                    save_both(pc_2cam_traj, pc_1cam_traj, img_traj, depth_traj,
                              state_traj, action_traj, base)
                keyboard._save_trajectory_flag = False

            if hasattr(keyboard, '_clear_trajectory_flag') and keyboard._clear_trajectory_flag:
                pc_2cam_traj.clear(); pc_1cam_traj.clear()
                img_traj.clear(); depth_traj.clear()
                state_traj.clear(); action_traj.clear()
                frame_count = 0
                keyboard._clear_trajectory_flag = False

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
        if pc_2cam_traj:
            logging.warning("紧急保存...")
            base = os.path.join(save_dir,
                f'{script_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_autosave.pkl')
            try:
                save_both(pc_2cam_traj, pc_1cam_traj, img_traj, depth_traj,
                          state_traj, action_traj, base)
            except Exception as e:
                logging.error(f"紧急保存失败: {e}")
        for cam_renderers in renderers.values():
            cam_renderers['rgb'].close()
            cam_renderers['depth'].close()
        env.close()
        logging.info("仿真结束")


if __name__ == "__main__":
    main()
