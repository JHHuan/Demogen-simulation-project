"""
Replay训练数据：直接执行训练数据中记录的actions

这个脚本会：
1. 从训练数据中读取所有episode的物体位置、agent位置和actions
2. 在仿真环境中重置到训练时的配置
3. 直接执行训练数据中的actions（不使用策略模型）
4. 统计成功率和失败案例

数据格式：
- action: (T, 7) - [x,y,z,qx,qy,qz,gripper] 绝对位置格式
- agent_pos: (T, 7) - [x,y,z,qx,qy,qz,gripper] 绝对位置格式

使用方法：
    cd /home/hjh/git_code/demogen/DemoGen-master/diffusion_policies
    python replay_training_data.py
"""

import numpy as np
import logging
import sys
import math
import mujoco
from pathlib import Path
from termcolor import cprint
import zarr

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

DIM_ACTION = 7          # cube任务的action维度（7维绝对位置）
N_POINTS = 1024         # 点云点数

# 任务相关参数
GRIPPER_OPEN_THRESH = 0.04  # 夹爪打开阈值
GRIPPER_CLOSE_THRESH = 0.02  # 夹爪关闭阈值
CUBE_HEIGHT = 0.03         # 立方体高度（米）
TABLE_HEIGHT = 0.42         # 桌面高度（米）


# ==========================================
# 数据加载函数
# ==========================================

def load_training_data(zarr_path):
    """加载训练数据并提取所有episode的信息"""
    cprint(f"Loading training data from {zarr_path}...", 'cyan')

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

            # Z轴过滤：选择高度在物体范围内的点
            z_mask = (xyz[:,2] >= 0.44) & (xyz[:,2] <= 0.48)
            xyz_filtered = xyz[z_mask]

            if len(xyz_filtered) > 0:
                # 使用中位数代替均值，更鲁棒（特别是对X方向的离群点）
                obj_center = np.median(xyz_filtered, axis=0)

                # 手动校准：补偿X方向的系统性偏差
                # 点云推断的X值通常比真实值大约0.03米
                obj_center[0] -= 0.03  # X方向减去偏差补偿

                # 提取这个episode的所有actions（包括最后一帧）
                # 因为采集时action和agent_pos相同，所以需要执行所有action
                ep_actions = actions[ep_start:ep_end]  # 从ep_start到ep_end的所有actions

                episodes_info.append({
                    'episode': ep_idx,
                    'start_idx': ep_start,
                    'length': ep_length,
                    'object_pos': obj_center,  # [x, y, z]
                    'agent_pos_step0': agent_pos[ep_start],  # Step 0的位置
                    'agent_pos_step1': [agent_step1[0], agent_step1[1], agent_step1[2]],  # Step 1的位置
                    'actions': ep_actions  # 训练数据中记录的actions
                })

    cprint(f"Loaded {len(episodes_info)} episodes", 'green')
    return episodes_info


# ==========================================
# 成功检测函数
# ==========================================

def check_task_success(final_agent_pos, object_lifted=False):
    """检查任务是否成功"""
    try:
        # 检查最后的夹爪状态
        final_gripper = final_agent_pos[6]

        # 检查最后的高度
        final_z = final_agent_pos[2]

        # 成功条件：夹爪闭合 + 抬升到一定高度
        gripper_closed = final_gripper < GRIPPER_CLOSE_THRESH
        lifted = final_z > 0.65  # 抬升到0.65以上

        if gripper_closed and lifted:
            return True, f"Success: gripper={final_gripper:.4f}, Z={final_z:.4f}"
        else:
            return False, f"Failed: gripper={final_gripper:.4f} (<{GRIPPER_CLOSE_THRESH}?), Z={final_z:.4f} (>0.65?)"
    except Exception as e:
        return False, f"Error checking success: {e}"


# ==========================================
# Episode执行函数
# ==========================================

def run_episode(env, episode_info):
    """运行单个episode，replay训练数据的actions"""
    ep_idx = episode_info['episode']
    object_pos = episode_info['object_pos']
    init_agent_pos = episode_info['agent_pos_step1']  # 使用Step 1的位置
    actions = episode_info['actions']

    cprint(f"\n{'='*60}", 'yellow')
    cprint(f"Episode {ep_idx}: Replaying {len(actions)} actions...", 'yellow')
    cprint(f"  Object pos: X={object_pos[0]:.4f}, Y={object_pos[1]:.4f}, Z={object_pos[2]:.4f} (从点云推断)", 'cyan')
    cprint(f"  Agent init:  X={init_agent_pos[0]:.4f}, Y={init_agent_pos[1]:.4f}, Z={init_agent_pos[2]:.4f}", 'cyan')
    cprint(f"{'='*60}", 'yellow')

    # 重置环境
    env.reset()
    env.controller.reference = 'world'

    # 设置物体位置
    # 使用从点云推断的位置（与采集时一致）
    object_pos_full = [object_pos[0], object_pos[1], object_pos[2]]
    object_quat = [1.0, 0.0, 0.0, 0.0]
    env.mj_data.joint('green_block:joint').qpos[:7] = object_pos_full + object_quat

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
        if z_error < 0.01:
            cprint(f"✓ Initialization converged in {init_step+1} steps", 'green')
            break

    # 执行actions
    agent_pos_history = []
    curr_quat_xyz = quat_xyz.copy()  # 当前quaternion xyz
    curr_quat_w = quat_w  # 当前quaternion w

    for action_idx, action in enumerate(actions):
        # 解析action（绝对位置格式：[x, y, z, qx, qy, qz, gripper]）
        target_pos = action[:3]  # [x, y, z] 目标位置的绝对坐标
        target_quat_xyz = action[3:6]  # [qx, qy, qz] 目标姿态
        target_grip = action[6]  # gripper的绝对值

        # 计算完整的quaternion（添加w分量）
        target_quat_w = np.sqrt(1 - np.sum(target_quat_xyz**2))
        full_quat = np.array([target_quat_w, target_quat_xyz[0], target_quat_xyz[1], target_quat_xyz[2]])

        # 构造目标action（CARTIK格式：7维 [x,y,z,qw,qx,qy,qz]）
        action_full = np.concatenate([target_pos, full_quat])

        # 控制夹爪（根据目标gripper值）
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
            if pos_error < pos_tolerance:
                break

        # 更新当前quaternion（用于记录）
        curr_quat_xyz = target_quat_xyz
        curr_quat_w = target_quat_w

        # 记录agent位置（7维：xyz + quat_xyz + gripper）
        # 注意：使用target_grip而不是物理仿真值，与采集时保持一致
        new_pos = env.robot.get_end_xpos()
        full_agent_pos = np.concatenate([new_pos, curr_quat_xyz, [target_grip]])  # 7维: [x,y,z,qx,qy,qz,grip]
        agent_pos_history.append(full_agent_pos)

        if action_idx % 10 == 0:
            logger.info(f"Action {action_idx}/{len(actions)}: pos=({new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f})")

    # 检查最终状态
    final_agent_pos = agent_pos_history[-1]
    success, reason = check_task_success(final_agent_pos)

    return success, reason, agent_pos_history


# ==========================================
# 主函数
# ==========================================

def main():
    env = None  # 初始化为None，便于finally块中使用

    try:
        # 加载训练数据
        script_dir = Path(__file__).parent.resolve()  # 绝对路径
        zarr_path = (script_dir.parent / "data" / "datasets" / "generated" / "0219-cube_test_81.zarr").resolve()
        cprint(f"Looking for zarr file at: {zarr_path}", 'cyan')
        if not zarr_path.exists():
            cprint(f"ERROR: Zarr file not found at {zarr_path}", 'red')
            return
        episodes_info = load_training_data(str(zarr_path))

        # 创建环境
        cprint("Creating MuJoCo environment...", 'cyan')
        env = RobotEnv(
            robot=PandaDemoGen,  # 传递类，不是实例
            render_mode='human',
            is_render_camera_offscreen=True,
            camera_in_render='demogen_camera',
            control_freq=100,
            controller='CARTIK',
        )

        # 运行所有episodes
        results = []

        cprint(f"\n{'='*80}", 'green')
        cprint(f"Starting replay of {len(episodes_info)} episodes", 'green')
        cprint(f"{'='*80}", 'green')

        for i, episode_info in enumerate(episodes_info):
            try:
                success, reason, agent_pos_history = run_episode(env, episode_info)
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
            for r in failed_results[:20]:  # 只显示前20个
                cprint(f"  Episode {r['episode']}: {r['reason']}", 'red')
            if len(failed_results) > 20:
                cprint(f"  ... and {len(failed_results) - 20} more", 'red')

        cprint(f"{'='*80}\n", 'yellow')

    finally:
        # 确保环境被正确关闭，避免GLX错误
        if env is not None:
            cprint("Closing environment...", 'cyan')
            try:
                env.close()
                cprint("Environment closed successfully.", 'green')
            except Exception as e:
                cprint(f"Warning: Error closing environment: {e}", 'yellow')


if __name__ == "__main__":
    main()
