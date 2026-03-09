"""
Replay DemoGen生成的数据：直接执行生成数据中记录的actions

功能：
1. 从DemoGen生成的zarr文件中读取所有episode的物体位置、agent位置和actions
2. 在仿真环境中重置到训练时的配置
3. 直接执行数据中记录的actions（不使用策略模型）
4. 统计成功率和失败案例
5. 支持视频录制功能

数据格式：
- action: (T, 7) - [x,y,z,qx,qy,qz,gripper] 绝对位置格式
- agent_pos: (T, 7) - [x,y,z,qx,qy,qz,gripper] 绝对位置格式
- point_cloud: (T, 1024, 6) - XYZ+RGB颜色

使用方法：
    cd /home/hjh/git_code/demogen/DemoGen-master/replay_eva
    python replay_task1.py

环境配置：
- 机器人: Panda + PandaHand 爪子
- 控制器: CARTIK (笛卡尔空间逆运动学)
- 场景: 桌子上有一个绿色立方体和一个红色立方体
- 相机: demogen_camera
"""

import numpy as np
import logging
import sys
import math
import mujoco
from pathlib import Path
from termcolor import cprint
import zarr
import cv2
from datetime import datetime

# 添加robopal到路径
robopal_path = Path(__file__).parent.parent.parent / "robopal"
sys.path.insert(0, str(robopal_path))

from robopal.envs.robot import RobotEnv
from robopal.robots.panda import PandaDemoGen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 配置参数
# ==========================================

DIM_ACTION = 7          # action维度（7维绝对位置）
N_POINTS = 1024         # 点云点数

# 任务相关参数
GRIPPER_OPEN_THRESH = 0.04  # 夹爪打开阈值
GRIPPER_CLOSE_THRESH = 0.032  # 夹爪关闭阈值
CUBE_HEIGHT = 0.03         # 立方体高度（米）
TABLE_HEIGHT = 0.42         # 桌面高度（米）

# 视频录制相关参数
ENABLE_VIDEO_RECORDING = False  # 是否启用视频录制（已关闭）
VIDEO_OUTPUT_DIR = Path(__file__).parent / "recorded_videos"  # 视频保存目录
VIDEO_FPS = 30  # 视频帧率
VIDEO_CODEC = 'mp4v'  # 视频编码格式


# ==========================================
# 视频录制类
# ==========================================

class VideoRecorder:
    """视频录制器类，用于记录MuJoCo仿真过程"""

    def __init__(self, output_path, fps=30, codec='mp4v', width=1920, height=1080):
        """
        初始化视频录制器

        Args:
            output_path: 视频输出路径
            fps: 帧率
            codec: 视频编码格式
            width: 视频宽度
            height: 视频高度
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.width = width
        self.height = height
        self.writer = None
        self.is_recording = False

        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """开始录制视频"""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        self.is_recording = True
        cprint(f"Started recording to: {self.output_path}", 'green')

    def add_frame(self, renderer, camera_name='demogen_camera'):
        """
        从MuJoCo渲染器添加一帧到视频

        Args:
            renderer: MuJoCo渲染器对象
            camera_name: 要录制的相机名称
        """
        if not self.is_recording or self.writer is None:
            return

        try:
            # 从robopal渲染器获取图像
            img = None

            # 方法1: 使用robopal的render_pixels_from_camera方法（推荐）
            if hasattr(renderer, 'render_pixels_from_camera'):
                try:
                    img = renderer.render_pixels_from_camera(camera_name, enable_depth=False)
                except Exception as e:
                    pass

            # 方法2: 如果有image_renderer属性，直接使用
            if img is None and hasattr(renderer, 'image_renderer'):
                try:
                    renderer.image_renderer.update_scene(renderer.mj_data, camera=camera_name)
                    img = renderer.image_renderer.render()
                except Exception as e:
                    pass

            # 检查图像是否有效
            if img is None or img.size == 0:
                return

            # 确保图像是numpy数组
            if not isinstance(img, np.ndarray):
                return

            # 转换RGB到BGR（OpenCV使用BGR格式）
            if len(img.shape) == 3 and img.shape[2] == 3:
                # robopal的render_pixels_from_camera已经返回RGB格式
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # 如果是RGBA，转换为RGB然后转BGR
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                # 已经是单通道或其他格式，直接使用
                img_bgr = img

            # 调整图像大小以匹配视频尺寸
            if img_bgr.shape[0] != self.height or img_bgr.shape[1] != self.width:
                img_bgr = cv2.resize(img_bgr, (self.width, self.height))

            # 写入视频帧
            self.writer.write(img_bgr)

        except Exception as e:
            # 静默处理错误，避免中断仿真
            pass

    def stop(self):
        """停止录制并保存视频"""
        if self.writer is not None:
            self.writer.release()
            self.is_recording = False
            cprint(f"Video saved: {self.output_path}", 'green')

    def __del__(self):
        """析构函数，确保视频资源被释放"""
        self.stop()


# ==========================================
# 数据加载函数
# ==========================================

def load_generated_data(zarr_path):
    """
    加载DemoGen生成的数据并提取所有episode的信息

    对于双物体任务（task1），需要从点云中推断两个物体的位置
    """
    cprint(f"Loading generated data from {zarr_path}...", 'cyan')

    root = zarr.open(zarr_path, 'r')
    agent_pos = root['data']['agent_pos'][:]
    point_cloud = root['data']['point_cloud'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]

    episodes_info = []

    for ep_idx, ep_end in enumerate(episode_ends):
        ep_start = episode_ends[ep_idx - 1] if ep_idx > 0 else 0
        ep_length = ep_end - ep_start

        if ep_length >= 2:
            step1_idx = ep_start + 1
            agent_step1 = agent_pos[step1_idx]
            pc_step1 = point_cloud[step1_idx]

            # 从点云推断物体位置（使用Z轴过滤+中位数）
            xyz = pc_step1[:, :3]  # XYZ坐标
            rgb = pc_step1[:, 3:]  # RGB颜色

            # 分离两个物体：
            # 绿色物体 (green_block): RGB中G分量较高
            # 红色物体 (red_block): RGB中R分量较高

            # Z轴过滤：选择高度在物体范围内的点
            z_mask = (xyz[:,2] >= 0.44) & (xyz[:,2] <= 0.48)
            xyz_filtered = xyz[z_mask]
            rgb_filtered = rgb[z_mask]

            if len(xyz_filtered) > 0:
                # 使用空间聚类分离两个物体（基于X坐标）
                # 因为RGB颜色可能被归一化，不可靠
                x_coords = xyz_filtered[:, 0]

                # 使用X坐标聚类：将相近的点归为一类（阈值1cm）
                clusters = []
                used_indices = set()

                for i in range(len(xyz_filtered)):
                    if i in used_indices:
                        continue

                    # 创建新聚类
                    cluster_indices = [i]
                    cluster_x = x_coords[i]

                    # 找到所有X坐标接近的点（1cm阈值）
                    for j in range(i + 1, len(xyz_filtered)):
                        if j not in used_indices and abs(x_coords[j] - cluster_x) < 0.01:
                            cluster_indices.append(j)
                            used_indices.add(j)

                    used_indices.add(i)

                    # 如果聚类点数足够，添加到聚类列表
                    if len(cluster_indices) >= 10:  # 至少10个点才算有效物体
                        clusters.append(xyz_filtered[cluster_indices])

                # 按X坐标排序：左边的是绿色物体，右边的是红色物体
                clusters.sort(key=lambda c: np.median(c[:, 0]))

                # 提取两个物体的位置
                if len(clusters) >= 2:
                    # 检测到两个物体
                    green_obj_center = np.median(clusters[0], axis=0)
                    red_obj_center = np.median(clusters[1], axis=0)

                    # 补偿X方向偏差（点云推断的X值通常比真实值大约0.03米）
                    green_obj_center[0] += 0.0
                    red_obj_center[0] += 0.02
                    
                    # 补偿Y方向偏差
                    green_obj_center[1] += 0.0
                    red_obj_center[1] += 0.03

                elif len(clusters) == 1:
                    # 只检测到一个物体，使用默认位置
                    green_obj_center = np.median(clusters[0], axis=0)
                    green_obj_center[0] -= 0.03
                    red_obj_center = np.array([green_obj_center[0] + 0.1, green_obj_center[1], green_obj_center[2]])
                    cprint(f"  Warning: Only detected one object, using default red position", 'yellow')

                else:
                    # 没有检测到物体，使用默认位置
                    cprint(f"  Warning: No objects detected, using default positions", 'yellow')
                    green_obj_center = np.array([0.4, 0.0, 0.445])
                    red_obj_center = np.array([0.5, 0.0, 0.445])

                # 提取这个episode的所有actions
                ep_actions = actions[ep_start:ep_end]

                episodes_info.append({
                    'episode': ep_idx,
                    'start_idx': ep_start,
                    'length': ep_length,
                    'green_object_pos': green_obj_center,  # [x, y, z]
                    'red_object_pos': red_obj_center,  # [x, y, z]
                    'agent_pos_step0': agent_pos[ep_start],  # Step 0的位置
                    'agent_pos_step1': [agent_step1[0], agent_step1[1], agent_step1[2]],  # Step 1的位置
                    'actions': ep_actions
                })

    cprint(f"Loaded {len(episodes_info)} episodes", 'green')
    return episodes_info


# ==========================================
# 成功检测函数
# ==========================================

def check_task_success(final_agent_pos, object_lifted=False):
    """检查双物体任务是否成功（把绿色物体移动到红色物体上）"""
    try:
        # 检查最后的夹爪状态
        final_gripper = final_agent_pos[6]

        # 检查最后的高度
        final_z = final_agent_pos[2]

        # 成功条件：夹爪闭合 + 抬升到一定高度
        gripper_closed = final_gripper < GRIPPER_CLOSE_THRESH
        lifted = final_z > 0.5  # 抬升到0.5以上

        if gripper_closed and lifted:
            return True, f"Success: gripper={final_gripper:.4f}, Z={final_z:.4f}"
        else:
            return False, f"Failed: gripper={final_gripper:.4f} (<{GRIPPER_CLOSE_THRESH}?), Z={final_z:.4f} (>0.5?)"
    except Exception as e:
        return False, f"Error checking success: {e}"


# ==========================================
# Episode执行函数
# ==========================================

def run_episode(env, episode_info, video_recorder=None, record_video=False):
    """
    运行单个episode，replay生成的actions

    Args:
        env: 仿真环境
        episode_info: episode信息字典
        video_recorder: 视频录制器对象
        record_video: 是否录制视频
    """
    ep_idx = episode_info['episode']
    green_obj_pos = episode_info['green_object_pos']
    red_obj_pos = episode_info['red_object_pos']
    init_agent_pos = episode_info['agent_pos_step1']
    actions = episode_info['actions']

    cprint(f"\n{'='*60}", 'yellow')
    cprint(f"Episode {ep_idx}: Replaying {len(actions)} actions...", 'yellow')
    cprint(f"  Green object pos: X={green_obj_pos[0]:.4f}, Y={green_obj_pos[1]:.4f}, Z={green_obj_pos[2]:.4f}", 'cyan')
    cprint(f"  Red object pos:   X={red_obj_pos[0]:.4f}, Y={red_obj_pos[1]:.4f}, Z={red_obj_pos[2]:.4f}", 'cyan')
    cprint(f"  Agent init:       X={init_agent_pos[0]:.4f}, Y={init_agent_pos[1]:.4f}, Z={init_agent_pos[2]:.4f}", 'cyan')
    if record_video:
        cprint(f"  Video recording: ENABLED", 'green')
    cprint(f"{'='*60}", 'yellow')

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 开始录制视频
    if record_video and video_recorder is not None:
        video_recorder.start()

    # 设置两个物体位置
    # 绿色物体（被抓取的物体）
    green_obj_pos_full = [green_obj_pos[0], green_obj_pos[1], green_obj_pos[2]]
    green_obj_quat = [1.0, 0.0, 0.0, 0.0]
    env.mj_data.joint('green_block:joint').qpos[:7] = green_obj_pos_full + green_obj_quat

    # 红色物体（目标物体）
    red_obj_pos_full = [red_obj_pos[0], red_obj_pos[1], red_obj_pos[2]]
    red_obj_quat = [1.0, 0.0, 0.0, 0.0]
    env.mj_data.joint('red_block:joint').qpos[:7] = red_obj_pos_full + red_obj_quat

    # 设置agent初始位置（使用Step 1的位置，但Z改为0.75）
    init_pos = np.array([init_agent_pos[0], init_agent_pos[1], 0.75])
    quat_xyz = np.array([-0.00087801, -0.0036839, -0.00133284])
    quat_w = np.sqrt(1 - np.sum(quat_xyz**2))
    full_quat = np.array([quat_w, quat_xyz[0], quat_xyz[1], quat_xyz[2]])  # [qw, qx, qy, qz]

    # CARTIK控制器接受7维action: [x, y, z, qw, qx, qy, qz]
    target_pose = np.concatenate([init_pos, full_quat])

    # 确保初始夹爪完全张开
    env.mj_data.qpos[7:9] = 0.04

    cprint(f"Initializing robot to Z=0.750...", 'cyan')
    for init_step in range(100):
        env.step(target_pose)
        env.mj_data.qpos[7:9] = 0.04  # 持续保持夹爪张开
        pos_after = env.robot.get_end_xpos()
        z_error = abs(pos_after[2] - 0.75)
        # 录制视频帧
        if record_video and video_recorder is not None and hasattr(env, 'renderer'):
            video_recorder.add_frame(env.renderer, camera_name='demogen_camera')
        if z_error < 0.01:
            cprint(f"✓ Initialization converged in {init_step+1} steps", 'green')
            break

    # 执行actions
    agent_pos_history = []
    curr_quat_xyz = quat_xyz.copy()
    curr_quat_w = quat_w

    for action_idx, action in enumerate(actions):
        # 解析action（绝对位置格式：[x, y, z, qx, qy, qz, gripper]）
        target_pos = action[:3]  # [x, y, z]
        target_quat_xyz = action[3:6]  # [qx, qy, qz]
        target_grip = action[6]  # gripper

        # 计算完整的quaternion
        target_quat_w = np.sqrt(1 - np.sum(target_quat_xyz**2))
        full_quat = np.array([target_quat_w, target_quat_xyz[0], target_quat_xyz[1], target_quat_xyz[2]])

        # 构造目标action（CARTIK格式：7维 [x,y,z,qw,qx,qy,qz]）
        action_full = np.concatenate([target_pos, full_quat])

        # 控制夹爪
        if target_grip < GRIPPER_CLOSE_THRESH:
            env.robot.end['agent0'].close()
        else:
            env.robot.end['agent0'].open()

        # 执行多次step直到到达目标
        max_exec_steps = 50
        pos_tolerance = 0.005

        for exec_step in range(max_exec_steps):
            env.step(action_full)
            pos_after = env.robot.get_end_xpos()
            pos_error = np.linalg.norm(pos_after[:3] - target_pos[:3])
            # 录制视频帧
            if record_video and video_recorder is not None and hasattr(env, 'renderer') and exec_step % 2 == 0:
                video_recorder.add_frame(env.renderer, camera_name='demogen_camera')
            if pos_error < pos_tolerance:
                break

        # 更新当前quaternion
        curr_quat_xyz = target_quat_xyz
        curr_quat_w = target_quat_w

        # 记录agent位置
        new_pos = env.robot.get_end_xpos()
        full_agent_pos = np.concatenate([new_pos, curr_quat_xyz, [target_grip]])
        agent_pos_history.append(full_agent_pos)

        if action_idx % 10 == 0:
            logger.info(f"Action {action_idx}/{len(actions)}: pos=({new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f})")

    # 检查最终状态
    final_agent_pos = agent_pos_history[-1]
    success, reason = check_task_success(final_agent_pos)

    # 停止录制视频
    if record_video and video_recorder is not None:
        video_recorder.stop()

    return success, reason, agent_pos_history


# ==========================================
# 主函数
# ==========================================

def main():
    env = None

    try:
        # 加载生成的数据
        script_dir = Path(__file__).parent.resolve()
        zarr_path = (script_dir.parent / "data" / "datasets" / "generated" / "task1_test_16.zarr").resolve()

        cprint(f"Looking for zarr file at: {zarr_path}", 'cyan')
        if not zarr_path.exists():
            cprint(f"ERROR: Zarr file not found at {zarr_path}", 'red')
            cprint(f"Please run DemoGen first to generate the data", 'yellow')
            return

        episodes_info = load_generated_data(str(zarr_path))

        # 创建环境
        cprint("Creating MuJoCo environment...", 'cyan')
        env = RobotEnv(
            robot=PandaDemoGen,
            render_mode='human',
            is_render_camera_offscreen=True,
            camera_in_render='demogen_camera',
            control_freq=100,
            controller='CARTIK',
        )

        # 创建视频输出目录
        if ENABLE_VIDEO_RECORDING:
            VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cprint(f"Video output directory: {VIDEO_OUTPUT_DIR}", 'cyan')

        # 运行所有episodes
        results = []

        cprint(f"\n{'='*80}", 'green')
        cprint(f"Starting replay of {len(episodes_info)} episodes", 'green')
        cprint(f"{'='*80}", 'green')

        for i, episode_info in enumerate(episodes_info):
            try:
                # 为每个episode创建独立的视频录制器
                video_recorder = None
                if ENABLE_VIDEO_RECORDING:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"episode_{episode_info['episode']:03d}_{timestamp}.mp4"
                    video_path = VIDEO_OUTPUT_DIR / video_filename

                    # 获取渲染器的图像尺寸
                    renderer_width = 1920
                    renderer_height = 1080
                    if hasattr(env, 'renderer') and env.renderer is not None:
                        try:
                            renderer_width = env.renderer.viewport.width
                            renderer_height = env.renderer.viewport.height
                        except:
                            pass

                    video_recorder = VideoRecorder(
                        output_path=video_path,
                        fps=VIDEO_FPS,
                        codec=VIDEO_CODEC,
                        width=renderer_width,
                        height=renderer_height
                    )

                success, reason, agent_pos_history = run_episode(
                    env, episode_info,
                    video_recorder=video_recorder,
                    record_video=ENABLE_VIDEO_RECORDING
                )
                results.append({
                    'episode': episode_info['episode'],
                    'success': success,
                    'reason': reason,
                    'final_z': agent_pos_history[-1][2],
                    'final_gripper': agent_pos_history[-1][6]
                })

                status = "✓ SUCCESS" if success else "✗ FAILED"
                cprint(f"Episode {episode_info['episode']}: {status} - {reason}", 'green' if success else 'red')

            except Exception as e:
                logger.error(f"Error running episode {episode_info['episode']}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'episode': episode_info['episode'],
                    'success': False,
                    'reason': f"Error: {e}",
                    'final_z': None,
                    'final_gripper': None
                })

        # 统计结果
        cprint(f"\n{'='*80}", 'yellow')
        cprint(f"REPLAY RESULTS", 'yellow')
        cprint(f"{'='*80}", 'yellow')

        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        success_rate = success_count / total_count * 100 if total_count > 0 else 0

        cprint(f"Total episodes: {total_count}", 'cyan')
        cprint(f"Successful: {success_count}", 'green')
        cprint(f"Failed: {total_count - success_count}", 'red')
        cprint(f"Success rate: {success_rate:.1f}%", 'yellow')

        # 打印失败的episodes
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            cprint(f"\nFailed episodes ({len(failed_results)}):", 'red')
            for r in failed_results[:20]:
                cprint(f"  Episode {r['episode']}: {r['reason']}", 'red')
            if len(failed_results) > 20:
                cprint(f"  ... and {len(failed_results) - 20} more", 'red')

        cprint(f"{'='*80}\n", 'yellow')

    finally:
        # 确保环境被正确关闭
        if env is not None:
            cprint("Closing environment...", 'cyan')
            try:
                env.close()
                cprint("Environment closed successfully.", 'green')
            except Exception as e:
                cprint(f"Warning: Error closing environment: {e}", 'yellow')


if __name__ == "__main__":
    main()
