"""
在MuJoCo仿真环境中部署和评估训练好的DP3策略

使用方法：
    cd /home/hjh/git_code/demogen/DemoGen-master/diffusion_policies
    python evaluate_sim.py checkpoint=../data/ckpts/0219-cube_test_225-dp3-seed0/checkpoints/521.ckpt

推理-执行模式（与真实机器人一致）:
    - 每次预测 N_ACTIONS 步动作
    - 每执行1步后，更新观测
    - 每 N_ACTIONS 步重新预测一次
    - 使用动作缓冲区存储预测的动作
"""

import numpy as np
import logging
import sys
import math
import mujoco
import torch
from pathlib import Path
from diffusion_policies.common.pytorch_util import dict_apply
import hydra
from omegaconf import OmegaConf
from termcolor import cprint

# 注册eval resolver
def register_eval_resolvers():
    """Register custom resolvers for OmegaConf"""
    def eval_resolver(s):
        return eval(s)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval_resolver)

register_eval_resolvers()

# 添加robopal到路径
robopal_path = Path(__file__).parent.parent.parent / "robopal"
sys.path.insert(0, str(robopal_path))

from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaDemoGen
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 配置参数（与real_world/evaluate.py一致）
# ==========================================

DIM_ACTION = 7          # cube任务的action维度：[x,y,z,qx,qy,qz,gripper]
N_POINTS = 1024         # 点云点数
HORIZON = 16            # 时间窗口
N_OBS = 2               # 观测步数
N_ACTIONS = 8           # 每次预测的动作步数
MAX_EPISODE_STEPS = 200 # 最大执行步数

# 任务相关参数
GRIPPER_OPEN_THRESH = 0.04  # 夹爪打开阈值
GRIPPER_CLOSE_THRESH = 0.031  # 夹爪闭合阈值
MIN_SAFE_Z = 0.4       # Z轴最小安全高度

OBS_KEYS = ['point_cloud', 'agent_pos']


# ==========================================
# 点云处理函数
# ==========================================

def depth_to_point_cloud(depth, rgb, camera_name, mj_model, mj_data):
    """将 MuJoCo 深度图转换为点云（xyz + rgb）"""
    height, width = depth.shape

    # 获取相机ID
    try:
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    except:
        for i in range(mj_model.ncam):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name == camera_name:
                cam_id = i
                break
        else:
            raise ValueError(f"Camera {camera_name} not found")

    # 相机内参
    fovy = mj_model.cam_fovy[cam_id]
    f = height / (2 * math.tan(math.radians(fovy) / 2))
    cx = width / 2
    cy = height / 2

    # 计算 3D 点坐标（相机坐标系）
    z = depth.astype(np.float32)
    z = np.clip(z, 0.02, 2.0)

    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    x = (u - cx) * z / f
    y = (v - cy) * z / f

    # 获取相机位姿
    cam_pos = mj_model.cam_pos[cam_id].copy()
    cam_mat = mj_model.cam_mat0[cam_id]
    cam_mat_3x3 = cam_mat.reshape(3, 3)

    rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points_cam_aligned = points_cam @ rot_x_180.T
    points_world = (cam_mat_3x3 @ points_cam_aligned.T).T + cam_pos

    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    point_cloud = np.concatenate([points_world, colors], axis=1)

    # 裁剪工作空间
    valid_mask = (
        np.isfinite(point_cloud[:, 0]) &
        np.isfinite(point_cloud[:, 1]) &
        np.isfinite(point_cloud[:, 2]) &
        (point_cloud[:, 2] > 0.423) &
        (point_cloud[:, 2] < 0.7) &
        (point_cloud[:, 0] > 0.1) &
        (point_cloud[:, 0] < 0.8) &
        (point_cloud[:, 1] > -0.5) &
        (point_cloud[:, 1] < 0.5)
    )
    point_cloud = point_cloud[valid_mask]

    return point_cloud


def preprocess_point_cloud(point_cloud, n_points=1024):
    """点云预处理：裁剪、采样到固定点数"""
    if point_cloud.shape[0] > n_points:
        indices = np.random.choice(point_cloud.shape[0], n_points, replace=False)
        point_cloud = point_cloud[indices]
    elif point_cloud.shape[0] < n_points:
        indices = np.random.choice(point_cloud.shape[0], n_points, replace=True)
        point_cloud = point_cloud[indices]

    return point_cloud


# ==========================================
# 环境回合运行
# ==========================================

def run_episode(policy, env, renderer, device, episode_idx=0, max_steps=MAX_EPISODE_STEPS):
    """运行一个回合 - 使用real_world/evaluate.py模式"""

    cprint(f"\n{'='*60}", 'cyan')
    cprint(f"Episode {episode_idx}", 'cyan')
    cprint(f"{'='*60}", 'cyan')

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 设置物体位置
    object_pos = [0.3, -0.1, 0.46]
    object_quat = [1.0, 0.0, 0.0, 0.0]
    env.mj_data.joint('green_block:joint').qpos[:7] = object_pos + object_quat

    # 初始位置
    init_pos = np.array([0.3, 0.0, 0.8])
    quat_xyz = np.array([-0.00087801, -0.0036839, -0.00133284])
    quat_w = np.sqrt(1 - np.sum(quat_xyz**2))
    full_quat = np.array([quat_w, quat_xyz[0], quat_xyz[1], quat_xyz[2]])
    target_pose = np.concatenate([init_pos, full_quat])
    env.mj_data.qpos[7:9] = 0.04

    cprint(f"Initializing robot to Z={init_pos[2]:.3f}...", 'yellow')

    # 初始化到起始位置
    for i in range(500):
        env.step(target_pose)
        env.mj_data.qpos[7:9] = 0.04
        current_pos = env.robot.get_end_xpos()
        pos_error = np.linalg.norm(current_pos[:3] - init_pos[:3])
        if pos_error < 0.01:
            cprint(f"✓ Initialization converged in {i} steps", 'green')
            break

    # ========== 初始化观测和动作缓冲区 ==========
    # 与 real_world/evaluate.py L68-75 一致
    all_obs_dict = {
        'point_cloud': np.zeros((max_steps, N_POINTS, 6)),
        'agent_pos': np.zeros((max_steps, 7))
    }
    all_actions = np.zeros((max_steps, DIM_ACTION))

    # 获取初始观测 (step 0)
    obs = get_observation(env, renderer)
    all_obs_dict['point_cloud'][0] = obs['point_cloud']
    all_obs_dict['agent_pos'][0] = obs['agent_pos']

    cprint("Starting episode with real_world pattern...", 'yellow')
    cprint(f"Pattern: Predict every {N_ACTIONS} steps", 'yellow')

    # ========== 主循环（从step 1开始）==========
    # 参考 real_world/evaluate.py L83-139
    action_idx = 1
    while action_idx < max_steps:
        if action_idx < 20:
            cprint(f"\n--- action_idx: {action_idx} ---", 'yellow')

        # ========== 1. 更新观测 ==========
        # real_world/evaluate.py L93-95
        obs = get_observation(env, renderer)
        all_obs_dict['point_cloud'][action_idx] = obs['point_cloud']
        all_obs_dict['agent_pos'][action_idx] = obs['agent_pos']

        # ========== 2. 周期性预测（每N_ACTIONS步预测一次）==========
        # real_world/evaluate.py L98: if action_idx % N_ACTIONS == 1
        if action_idx % N_ACTIONS == 1:
            # 准备观测：使用最近N_OBS步
            # real_world/evaluate.py L105-108
            np_obs_dict = {
                'point_cloud': all_obs_dict['point_cloud'][action_idx-N_OBS+1:action_idx+1],
                'agent_pos': all_obs_dict['agent_pos'][action_idx-N_OBS+1:action_idx+1]
            }

            try:
                # 转换为torch tensor
                # real_world/evaluate.py L112-117
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))
                obs_dict_input = {}
                for key in OBS_KEYS:
                    obs_dict_input[key] = obs_dict[key].unsqueeze(0)

                # 模型推理
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                # 填充动作缓冲区
                # real_world/evaluate.py L122
                all_actions[action_idx:action_idx+N_ACTIONS] = np.squeeze(np_action_dict['action'])

                if action_idx < N_ACTIONS + 5:
                    cprint(f"✓ Predicted actions for steps {action_idx}:{action_idx+N_ACTIONS}", 'green')
                    for i in range(min(3, N_ACTIONS)):
                        act = all_actions[action_idx + i]
                        direction = "UP" if act[2] > obs['agent_pos'][2] else "DOWN"
                        cprint(f"  Action[{action_idx+i}]: pos=[{act[0]:+.4f},{act[1]:+.4f},{act[2]:.4f}] {direction}, grip={act[6]:.4f}", 'cyan')

            except Exception as e:
                cprint(f"✗ Prediction failed: {e}", 'red')
                import traceback
                traceback.print_exc()
                return False, f"Prediction error: {e}", action_idx

        # ========== 3. 从缓冲区取动作执行 ==========
        # real_world/evaluate.py L134-137
        action_todo = all_actions[action_idx]

        # 安全检查
        if action_todo[2] < MIN_SAFE_Z:
            if action_idx < 20:
                cprint(f"  ⚠️ Safety: Z={action_todo[2]:.4f} < MIN_SAFE_Z, clipping to {MIN_SAFE_Z}", 'yellow')
            action_todo[2] = MIN_SAFE_Z

        # 检测NaN/Inf
        if np.any(np.isnan(action_todo)) or np.any(np.isinf(action_todo)):
            cprint(f"✗ Warning: NaN/Inf at action_idx {action_idx}, skipping", 'red')
            action_idx += 1
            continue

        # 解析动作
        action_pos = action_todo[:3]
        euler_angles = action_todo[3:6]
        gripper = action_todo[6]

        # 计算四元数
        qx, qy, qz = euler_angles
        qw_squared = np.clip(1.0 - (qx**2 + qy**2 + qz**2), 0.0, 1.0)
        qw = np.sqrt(qw_squared)
        quat = np.array([qw, qx, qy, qz])

        action_full = np.concatenate([action_pos, quat])

        # 设置夹爪
        if gripper < GRIPPER_CLOSE_THRESH:
            env.robot.end['agent0'].close()
        else:
            env.robot.end['agent0'].open()

        # 执行动作
        max_exec_steps = 50
        min_exec_steps = 10
        pos_tolerance = 0.001

        for exec_step in range(max_exec_steps):
            env.step(action_full)
            if exec_step >= min_exec_steps:
                pos_after = env.robot.get_end_xpos()
                pos_error = np.linalg.norm(pos_after[:3] - action_pos[:3])
                if pos_error < pos_tolerance:
                    break

        if action_idx < 20:
            actual_error = np.linalg.norm(pos_after[:3] - action_pos[:3])
            current_pos_before = obs['agent_pos'][:3]
            direction = "UP" if action_pos[2] > current_pos_before[2] else "DOWN"
            cprint(f"  Executed action[{action_idx}]: pos=[{action_pos[0]:+.4f},{action_pos[1]:+.4f},{action_pos[2]:.4f}] {direction}, error={actual_error:.4f}", 'cyan')

        action_idx += 1

    cprint(f"Episode completed after {action_idx} steps", 'cyan')
    return True, "Completed", action_idx


def get_observation(env, renderer):
    """获取当前观测（点云 + 机器人状态）"""
    # 渲染深度图和RGB图像
    renderer.update_scene(env.mj_data, camera='demogen_camera')
    depth_image = renderer.render()

    rgb_renderer = mujoco.Renderer(env.mj_model)
    rgb_renderer.update_scene(env.mj_data, camera='demogen_camera')
    rgb_image = rgb_renderer.render()
    rgb_image = rgb_image[:, :, ::-1]

    # 生成点云
    point_cloud_raw = depth_to_point_cloud(
        depth_image, rgb_image, 'demogen_camera',
        env.mj_model, env.mj_data
    )
    point_cloud = preprocess_point_cloud(point_cloud_raw, n_points=N_POINTS)

    # 获取机器人状态
    end_effector_pos = env.robot.get_end_xpos()
    end_effector_quat = env.robot.get_end_xquat()
    r = R.from_quat([end_effector_quat[1], end_effector_quat[2],
                    end_effector_quat[3], end_effector_quat[0]])
    euler = r.as_euler('xyz', degrees=False)
    gripper_joints = env.mj_data.qpos[7:9]

    agent_pos = np.concatenate([
        end_effector_pos, euler, gripper_joints[:1]
    ])

    return {
        'point_cloud': point_cloud,
        'agent_pos': agent_pos
    }


@hydra.main(
    config_path="diffusion_policies/config",
    config_name="dp3",
    version_base="1.1"
)
def main(cfg: OmegaConf):
    # 强制使用cube任务配置
    original_cwd = Path(hydra.utils.get_original_cwd())
    cube_config_path = original_cwd / "diffusion_policies" / "config" / "task" / "cube.yaml"

    if not cube_config_path.exists():
        raise FileNotFoundError(f"Cannot find cube.yaml at {cube_config_path}")

    cfg.task = OmegaConf.load(cube_config_path)
    cfg.task_name = cfg.task.name
    cfg.shape_meta = cfg.task.shape_meta
    OmegaConf.resolve(cfg)

    # 从hydra参数获取checkpoint路径
    checkpoint_str = "../data/ckpts/0216-cube_fixed-dp3-seed0/checkpoints/331.ckpt"
    for arg in sys.argv:
        if arg.startswith("checkpoint="):
            checkpoint_str = arg.split("=")[1]
            break

    if not Path(checkpoint_str).is_absolute():
        checkpoint_path = Path(hydra.utils.get_original_cwd()) / checkpoint_str
    else:
        checkpoint_path = Path(checkpoint_str)

    if not checkpoint_path.is_file():
        cprint(f"Checkpoint not found: {checkpoint_path}", 'red')
        return

    cprint(f"Loading checkpoint from {checkpoint_path}", 'magenta')

    from diffusion_policies.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace

    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(cfg)
    model = workspace.model

    workspace.load_checkpoint(path=checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.use_ema:
        cprint("Using EMA model for inference", 'yellow')
        policy = workspace.ema_model.to(device)
    else:
        cprint("Using regular model for inference", 'yellow')
        policy = workspace.model.to(device)
    policy.eval()

    cprint(f"Policy loaded successfully on {device}", 'green')

    # 创建MuJoCo环境
    cprint("Creating MuJoCo environment...", 'cyan')
    env = RobotEnv(
        robot=PandaDemoGen,
        render_mode='human',
        is_render_camera_offscreen=True,
        camera_in_render='demogen_camera',
        control_freq=100,
        controller='CARTIK',
    )

    renderer = mujoco.Renderer(env.mj_model)
    renderer.enable_depth_rendering()

    # 运行回合
    n_episodes = cfg.get("n_episodes", 1)
    success_count = 0
    results = []

    for episode_idx in range(n_episodes):
        success, reason, steps = run_episode(
            policy, env, renderer, device,
            episode_idx=episode_idx
        )

        results.append({
            'episode': episode_idx,
            'success': success,
            'reason': reason,
            'steps': steps
        })

        if success:
            success_count += 1

        status = "✓ Success" if success else "✗ Failed"
        cprint(f"Episode {episode_idx}: {status} ({reason}) - {steps} steps",
               'green' if success else 'red')

    # 统计结果
    cprint(f"\n{'='*60}", 'cyan')
    cprint(f"Evaluation Summary", 'cyan')
    cprint(f"{'='*60}", 'cyan')
    cprint(f"Total episodes: {n_episodes}", 'white')
    cprint(f"Success count: {success_count}", 'white')
    if n_episodes > 0:
        cprint(f"Success rate: {success_count / n_episodes * 100:.1f}%", 'white')

    for result in results:
        status = "✓" if result['success'] else "✗"
        cprint(f"  Episode {result['episode']}: {status} - {result['reason']} ({result['steps']} steps)",
               'green' if result['success'] else 'red')

    cprint(f"{'='*60}\n", 'cyan')


if __name__ == '__main__':
    main()
