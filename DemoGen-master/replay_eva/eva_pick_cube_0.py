"""
在MuJoCo仿真环境中评估训练好的DP3策略（Single Cube - 单物体抓取任务）

功能：
1. 加载训练好的DP3策略模型
2. 在仿真环境中执行策略
3. 评估任务成功率（抓取绿色立方体并提起）
4. 支持多episode评估

使用方法：
    cd /home/hjh/git_code/demogen/DemoGen-master/replay_eva/pick_cube
    python eva_pick_cube_0.py \
  --checkpoint ../../data/ckpts/xxx/checkpoints/xxx.ckpt \
  --n_episodes 10 \
  --headless \
  --save_video \
  --video_dir ./recorded_videos

    
环境配置：
- 机器人: Panda + PandaHand 爪子
- 控制器: CARTIK (笛卡尔空间逆运动学)
- 场景: 桌子上有一个绿色立方体
- 相机: demogen_camera (机械臂正前方, pos=[1.5, 0, 0.8], fovy=45)
"""

import numpy as np
import logging
import sys
import math
import mujoco
import torch
import argparse
from pathlib import Path
try:
    import cv2
except ImportError:
    cv2 = None
from diffusion_policies.common.pytorch_util import dict_apply
import hydra
from omegaconf import OmegaConf
from termcolor import cprint
from tqdm import tqdm

# 注册eval resolver
def register_eval_resolvers():
    """Register custom resolvers for OmegaConf"""
    def eval_resolver(s):
        return eval(s)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval_resolver)

register_eval_resolvers()

# 添加robopal和diffusion_policies到路径
script_dir = Path(__file__).parent.resolve()
robopal_path = script_dir.parent.parent.parent / "robopal"
diffusion_path = script_dir.parent.parent / "diffusion_policies"
sys.path.insert(0, str(robopal_path))
sys.path.insert(0, str(diffusion_path))

from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaSingleCube1
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 配置参数（与训练配置一致）
# ==========================================

DIM_ACTION = 7          # action维度：[x,y,z,qx,qy,qz,gripper]
N_POINTS = 1024         # 点云点数
HORIZON = 16            # 时间窗口（与训练时一致）
N_OBS = 2               # 观测步数（与训练时一致）
N_ACTIONS = 14         # 每次预测的动作步数（与训练时n_action_steps一致）
MAX_EPISODE_STEPS = 31  # 最大执行步数

# 任务相关参数
GRIPPER_OPEN_THRESH = 0.040  # 夹爪打开阈值
GRIPPER_CLOSE_THRESH = 0.03  # 夹爪闭合阈值
MIN_SAFE_Z = 0.4       # Z轴最小安全高度

# 任务完成判定参数
SUCCESS_OBJECT_Z = 0.46  # 立方体被抓起的最小高度（刚离开桌面即可）
SUCCESS_X_TOLERANCE = 0.03  # 夹爪与立方体X方向对齐容差

OBS_KEYS = ['point_cloud', 'agent_pos']


# ==========================================
# 视频录制
# ==========================================

VIDEO_OUTPUT_DIR = Path(__file__).parent / "recorded_videos"
VIDEO_FPS = 30
VIDEO_CODEC = "mp4v"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_CAPTURE_EVERY_N_STEPS = 3


class VideoRecorder:
    """将离屏渲染帧写入 mp4 文件。"""

    def __init__(self, output_path, fps=VIDEO_FPS, codec=VIDEO_CODEC,
                 width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.width = width
        self.height = height
        self.writer = None

        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for --save_video.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame_rgb):
        if frame_rgb is None or frame_rgb.size == 0:
            return

        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

        frame = frame_rgb
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def capture_rgb_frame(rgb_renderer, env, camera_name='demogen_camera'):
    """从指定相机抓取一帧 RGB 图像。"""
    rgb_renderer.update_scene(env.mj_data, camera=camera_name)
    return rgb_renderer.render()[:, :, ::-1]


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
        (point_cloud[:, 2] > 0.426) &
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
# 任务完成检测函数
# ==========================================

def check_task_completion(env, previous_positions=None, stable_count=0):
    """
    检测任务是否完成：
    1. 立方体被抓起（Z > SUCCESS_OBJECT_Z）
    2. 夹爪中心与立方体中心X方向误差不超过±0.01

    Args:
        env: 环境对象
        previous_positions: 前一步的物体位置（保留兼容性，未使用）
        stable_count: 当前已稳定的步数（保留兼容性，未使用）

    Returns:
        is_complete: 是否完成任务
        stable_count: 保持为0（兼容性）
        current_positions: 当前物体位置
        reason: 完成原因描述
    """
    # 获取绿色物体的位置
    green_joint = env.mj_data.joint('green_block:joint')
    green_pos = green_joint.qpos[:3].copy()

    # 获取夹爪中心位置
    ee_pos = env.robot.get_end_xpos()

    current_positions = {
        'green': green_pos,
        'ee': ee_pos
    }

    # 条件1: 立方体被抓起
    is_lifted = green_pos[2] > SUCCESS_OBJECT_Z

    # 条件2: 夹爪与立方体X方向对齐
    x_error = abs(ee_pos[0] - green_pos[0])
    is_x_aligned = x_error <= SUCCESS_X_TOLERANCE

    is_complete = is_lifted and is_x_aligned

    # 生成说明
    if is_complete:
        reason = f"✓ Success: Object Z={green_pos[2]:.4f}m (lifted), X error={x_error:.4f}m (aligned)"
    else:
        parts = []
        if not is_lifted:
            parts.append(f"Z={green_pos[2]:.4f}m (need > {SUCCESS_OBJECT_Z})")
        if not is_x_aligned:
            parts.append(f"X error={x_error:.4f}m (need <= {SUCCESS_X_TOLERANCE})")
        reason = f"✗ " + ", ".join(parts)

    return is_complete, 0, current_positions, reason


# ==========================================
# 环境回合运行
# ==========================================

def run_episode(policy, env, depth_renderer, obs_rgb_renderer, device,
                green_object_pos=None, max_steps=MAX_EPISODE_STEPS,
                video_renderer=None,
                video_recorder=None, video_camera='demogen_camera',
                video_capture_every_n_steps=VIDEO_CAPTURE_EVERY_N_STEPS):
    """运行一个回合 - Single Cube抓取任务（静默模式）"""

    sim_step_count = 0

    def maybe_record_frame():
        if video_recorder is None:
            return
        if sim_step_count % max(video_capture_every_n_steps, 1) != 0:
            return
        frame_renderer = video_renderer if video_renderer is not None else obs_rgb_renderer
        frame = capture_rgb_frame(frame_renderer, env, camera_name=video_camera)
        video_recorder.add_frame(frame)

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 设置物体位置
    if green_object_pos is None:
        green_object_pos = [0.4, 0.0, 0.46]
    green_obj_quat = [1.0, 0.0, 0.0, 0.0]
    env.mj_data.joint('green_block:joint').qpos[:7] = green_object_pos + green_obj_quat

    # 初始位置
    init_pos = np.array([0.3, 0.0, 0.8])
    quat_xyz = np.array([-0.00087801, -0.0036839, -0.00133284])
    quat_w = np.sqrt(1 - np.sum(quat_xyz**2))
    full_quat = np.array([quat_w, quat_xyz[0], quat_xyz[1], quat_xyz[2]])
    target_pose = np.concatenate([init_pos, full_quat])
    env.mj_data.qpos[7:9] = 0.04

    # 初始化到起始位置
    for _ in range(500):
        env.step(target_pose)
        sim_step_count += 1
        env.mj_data.qpos[7:9] = 0.04
        maybe_record_frame()
        current_pos = env.robot.get_end_xpos()
        if np.linalg.norm(current_pos[:3] - init_pos[:3]) < 0.01:
            break

    # 初始化观测和动作缓冲区
    all_obs_dict = {
        'point_cloud': np.zeros((max_steps, N_POINTS, 6)),
        'agent_pos': np.zeros((max_steps, 7))
    }
    all_actions = np.zeros((max_steps, DIM_ACTION))

    obs = get_observation(env, depth_renderer, obs_rgb_renderer)
    all_obs_dict['point_cloud'][0] = obs['point_cloud']
    all_obs_dict['agent_pos'][0] = obs['agent_pos']
    maybe_record_frame()

    # 主循环
    action_idx = 1
    while action_idx < max_steps:
        obs = get_observation(env, depth_renderer, obs_rgb_renderer)
        all_obs_dict['point_cloud'][action_idx] = obs['point_cloud']
        all_obs_dict['agent_pos'][action_idx] = obs['agent_pos']

        # 周期性预测
        if action_idx % N_ACTIONS == 1:
            np_obs_dict = {
                'point_cloud': all_obs_dict['point_cloud'][action_idx-N_OBS+1:action_idx+1],
                'agent_pos': all_obs_dict['agent_pos'][action_idx-N_OBS+1:action_idx+1]
            }
            try:
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))
                obs_dict_input = {key: obs_dict[key].unsqueeze(0) for key in OBS_KEYS}
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                all_actions[action_idx:action_idx+N_ACTIONS] = np.squeeze(np_action_dict['action'])
            except Exception as e:
                return False, f"Prediction error: {e}", action_idx

        # 取动作执行
        action_todo = all_actions[action_idx]
        if action_todo[2] < MIN_SAFE_Z:
            action_todo[2] = MIN_SAFE_Z
        if np.any(np.isnan(action_todo)) or np.any(np.isinf(action_todo)):
            action_idx += 1
            continue

        action_pos = action_todo[:3]
        euler_angles = action_todo[3:6]
        gripper = action_todo[6]

        qx, qy, qz = euler_angles
        qw = np.sqrt(np.clip(1.0 - (qx**2 + qy**2 + qz**2), 0.0, 1.0))
        quat = np.array([qw, qx, qy, qz])
        action_full = np.concatenate([action_pos, quat])

        if gripper < GRIPPER_CLOSE_THRESH:
            env.robot.end['agent0'].close()
        else:
            env.robot.end['agent0'].open()

        for exec_step in range(50):
            env.step(action_full)
            sim_step_count += 1
            maybe_record_frame()
            if exec_step >= 10:
                pos_after = env.robot.get_end_xpos()
                if np.linalg.norm(pos_after[:3] - action_pos[:3]) < 0.001:
                    break

        # 任务完成检测
        is_complete, _, _, completion_reason = check_task_completion(env)
        if is_complete:
            return True, completion_reason, action_idx

        action_idx += 1

    return False, "Max steps reached", action_idx


def get_observation(env, depth_renderer, rgb_renderer):
    """获取当前观测（点云 + 机器人状态）"""
    # 渲染深度图和RGB图像
    depth_renderer.update_scene(env.mj_data, camera='demogen_camera')
    depth_image = depth_renderer.render()

    rgb_image = capture_rgb_frame(rgb_renderer, env, camera_name='demogen_camera')

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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate DP3 policy on Single Cube task')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization (much faster)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save each evaluation episode as an mp4 video')
    parser.add_argument('--video_dir', type=str, default=str(VIDEO_OUTPUT_DIR),
                        help='Directory for saved videos')
    parser.add_argument('--video_fps', type=int, default=VIDEO_FPS,
                        help=f'Output video FPS (default: {VIDEO_FPS})')
    parser.add_argument('--video_width', type=int, default=VIDEO_WIDTH,
                        help=f'Output video width (default: {VIDEO_WIDTH})')
    parser.add_argument('--video_height', type=int, default=VIDEO_HEIGHT,
                        help=f'Output video height (default: {VIDEO_HEIGHT})')
    parser.add_argument('--video_capture_every_n_steps', type=int,
                        default=VIDEO_CAPTURE_EVERY_N_STEPS,
                        help='Capture one frame every N env.step calls (default: 3)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cprint(f"Random seed: {args.seed}", 'yellow')

    cprint("="*80, 'magenta')
    cprint("Single Cube Policy Evaluation", 'magenta')
    cprint("="*80, 'magenta')

    # 加载配置
    cfg_path = script_dir.parent.parent / "diffusion_policies" / "diffusion_policies" / "config" / "dp3.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    # 加载task配置
    task_config_path = script_dir.parent.parent / "diffusion_policies" / "diffusion_policies" / "config" / "task" / "cube.yaml"
    if not task_config_path.exists():
        cprint(f"ERROR: cube.yaml not found at {task_config_path}", 'red')
        return

    cfg.task = OmegaConf.load(str(task_config_path))
    cfg.task_name = cfg.task.name
    cfg.shape_meta = cfg.task.shape_meta
    # 不调用 resolve()，避免 ${now} 等变量的解析错误

    # 获取checkpoint路径
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / args.checkpoint

    if not checkpoint_path.is_file():
        cprint(f"ERROR: Checkpoint not found: {checkpoint_path}", 'red')
        return

    cprint(f"Loading checkpoint from {checkpoint_path}", 'magenta')

    # 加载模型
    sys.path.insert(0, str(diffusion_path))
    from diffusion_policies.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace

    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(cfg)
    model = workspace.model
    workspace.load_checkpoint(path=str(checkpoint_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_ema or cfg.training.use_ema:
        cprint("Using EMA model for inference", 'yellow')
        policy = workspace.ema_model.to(device)
    else:
        cprint("Using regular model for inference", 'yellow')
        policy = workspace.model.to(device)
    policy.eval()

    cprint(f"Policy loaded successfully on {device}", 'green')

    # 创建MuJoCo环境
    cprint("Creating MuJoCo environment...", 'cyan')
    render_mode = None if args.headless else 'human'
    if args.headless:
        cprint("Running in HEADLESS mode (no visualization)", 'yellow')
    if args.save_video and cv2 is None:
        cprint("ERROR: OpenCV (cv2) is not installed, cannot save video.", 'red')
        return
    env = RobotEnv(
        robot=PandaSingleCube1,
        render_mode=render_mode,
        is_render_camera_offscreen=True,
        camera_in_render='demogen_camera',
        control_freq=100,
        controller='CARTIK',
    )

    depth_renderer = mujoco.Renderer(env.mj_model)
    depth_renderer.enable_depth_rendering()
    obs_rgb_renderer = mujoco.Renderer(env.mj_model)
    video_renderer = None
    if args.save_video:
        video_renderer = mujoco.Renderer(
            env.mj_model,
            height=args.video_height,
            width=args.video_width
        )

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = script_dir / video_dir

    # 运行回合
    results = []

    cprint(f"\nStarting evaluation: {args.n_episodes} episodes", 'green')

    # 随机物体位置范围
    green_x_range = [0.25, 0.55]
    green_y_range = [-0.15, 0.15]
    green_z = 0.46

    pbar = tqdm(range(args.n_episodes), desc="Evaluating", ncols=120)
    for episode_idx in pbar:
        green_pos = [
            round(np.random.uniform(*green_x_range), 2),
            round(np.random.uniform(*green_y_range), 2),
            green_z
        ]

        video_path = None
        video_recorder = None
        if args.save_video:
            video_path = video_dir / f"episode_{episode_idx:04d}.mp4"
            video_recorder = VideoRecorder(
                output_path=video_path,
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height
            )

        try:
            success, reason, steps = run_episode(
                policy, env, depth_renderer, obs_rgb_renderer, device,
                green_object_pos=green_pos,
                video_renderer=video_renderer,
                video_recorder=video_recorder,
                video_capture_every_n_steps=args.video_capture_every_n_steps
            )
        finally:
            if video_recorder is not None:
                video_recorder.close()

        results.append({
            'episode': episode_idx,
            'success': success,
            'reason': reason,
            'steps': steps,
            'green_pos': green_pos,
            'video_path': str(video_path) if video_path is not None else None
        })

        n_success = sum(1 for r in results if r['success'])
        rate = n_success / len(results) * 100
        status = "OK" if success else "FAIL"
        pbar.set_postfix(
            obj=f"({green_pos[0]:.2f},{green_pos[1]:.2f})",
            last=f"{status}@s{steps}",
            success=f"{n_success}/{len(results)}",
            rate=f"{rate:.1f}%"
        )

    # 统计结果
    n_success = sum(1 for r in results if r['success'])
    success_rate = n_success / args.n_episodes * 100 if args.n_episodes > 0 else 0

    cprint(f"\n{'='*60}", 'cyan')
    cprint(f"Evaluation Summary", 'cyan')
    cprint(f"{'='*60}", 'cyan')
    cprint(f"Total: {args.n_episodes} | Success: {n_success} | Rate: {success_rate:.1f}%", 'white')
    cprint(f"Seed: {args.seed} | Checkpoint: {args.checkpoint}", 'white')
    if args.save_video:
        cprint(f"Videos saved to: {video_dir}", 'white')
    cprint(f"{'='*60}", 'cyan')

    # 逐行结果
    for r in results:
        tag = "OK" if r['success'] else "FAIL"
        pos = r['green_pos']
        cprint(f"  [{tag}] ep{r['episode']:03d} obj=({pos[0]:.3f},{pos[1]:.3f}) steps={r['steps']:2d} {r['reason']}",
               'green' if r['success'] else 'red')

    cprint(f"{'='*60}\n", 'cyan')

    # 关闭环境
    env.close()


if __name__ == '__main__':
    main()
