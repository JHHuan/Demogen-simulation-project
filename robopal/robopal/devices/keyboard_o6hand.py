"""
O6Hand灵巧手专用键盘控制器
扩展了基础Keyboard类，增加数字键控制灵巧手关节的功能
"""
import time
import logging
import numpy as np

try:
    from pynput import keyboard
except ImportError:
    logging.warn("pynput is not installed. Please install it by running 'pip install pynput'")

import robopal.commons.transform as T
from robopal.devices import BaseDevice


class O6HandKeyboard(BaseDevice):
    """支持O6Hand灵巧手控制的键盘类"""

    def __init__(self, pos_scale=0.01, rot_scale=0.01) -> None:
        super().__init__(pos_scale, rot_scale)

        self._is_ctrl_l_pressed = False
        self._is_shift_pressed = False
        self._end_pos_offset = np.array([0.0, 0.0, 0.0])
        self._end_rot_offset = np.eye(3)
        self._record_frame_flag = False
        self._save_trajectory_flag = False
        self._clear_trajectory_flag = False

        # O6Hand灵巧手控制标志
        self._hand_joint_flags = [False] * 6  # 6个关节的控制标志
        self._hand_open_all_flag = False  # 全部张开标志
        self._hand_close_all_flag = False  # 全部闭合标志
        self._hand_grasp_four_flag = False  # 四指闭合标志
        self._hand_release_four_flag = False  # 四指张开标志

    def start(self):
        self.command_introduction()

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()

    def command_introduction(self):
        logging.info("\n" + "=" * 60)
        logging.info("O6Hand灵巧手键盘控制说明")
        logging.info("=" * 60)
        logging.info("【末端执行器控制】")
        logging.info("  方向键 ↑/↓/←/→     : 控制末端在 x/y 平面移动")
        logging.info("  Ctrl + ↑/↓         : 控制 z 轴移动 (↑上升, ↓下降)")
        logging.info("  Shift + 方向键      : 旋转末端执行器")
        logging.info("")
        logging.info("【灵巧手控制】")
        logging.info("  数字键 1           : 大拇指外展/内收 (thumb_yaw)")
        logging.info("  数字键 2           : 大拇指屈曲 (thumb_pitch)")
        logging.info("  数字键 3           : 食指屈曲 (index)")
        logging.info("  数字键 4           : 中指屈曲 (middle)")
        logging.info("  数字键 5           : 无名指屈曲 (ring)")
        logging.info("  数字键 6           : 小指屈曲 (pinky)")
        logging.info("  Shift + 数字      : 反向运动")
        logging.info("  o 键              : 灵巧手全部张开")
        logging.info("  c 键              : 灵巧手全部闭合")
        logging.info("  g 键              : 四指闭合（快捷抓取，自动录制）")
        logging.info("  f 键              : 四指张开（快捷释放，自动录制）")
        logging.info("")
        logging.info("【其他功能】")
        logging.info("  空格键            : 手动录制一帧")
        logging.info("  r 键              : 重置环境")
        logging.info("  a 键              : 保存轨迹")
        logging.info("  x 键              : 清除轨迹")
        logging.info("  ESC               : 退出")
        logging.info("=" * 60 + "\n")

    def on_press(self, key):
        try:
            # === 方向键控制 ===
            if key == keyboard.Key.up:
                if self._is_ctrl_l_pressed:
                    if self._is_shift_pressed:
                        self._end_rot_offset = self._end_rot_offset.dot(
                            T.euler_2_mat(self.rot_scale * np.array([0, 0, 1]))
                        )
                    else:
                        self._end_pos_offset[2] += self.pos_scale
                elif self._is_shift_pressed:
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self.rot_scale * np.array([0, -1, 0]))
                    )
                else:
                    self._end_pos_offset[0] -= self.pos_scale

            elif key == keyboard.Key.down:
                if self._is_ctrl_l_pressed:
                    if self._is_shift_pressed:
                        self._end_rot_offset = self._end_rot_offset.dot(
                            T.euler_2_mat(self.rot_scale * np.array([0, 0, -1]))
                        )
                    else:
                        self._end_pos_offset[2] -= self.pos_scale
                elif self._is_shift_pressed:
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self.rot_scale * np.array([0, 1, 0]))
                    )
                else:
                    self._end_pos_offset[0] += self.pos_scale

            elif key == keyboard.Key.left:
                if self._is_shift_pressed:
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self.rot_scale * np.array([1, 0, 0]))
                    )
                else:
                    self._end_pos_offset[1] -= self.pos_scale

            elif key == keyboard.Key.right:
                if self._is_shift_pressed:
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self.rot_scale * np.array([-1, 0, 0]))
                    )
                else:
                    self._end_pos_offset[1] += self.pos_scale

            # === 修饰键 ===
            elif key == keyboard.Key.ctrl_l:
                self._is_ctrl_l_pressed = True

            elif key == keyboard.Key.shift:
                self._is_shift_pressed = True

            # === 功能键 ===
            elif key == keyboard.Key.space:
                self._record_frame_flag = True

            # === 处理数字键和字符键 ===
            else:
                try:
                    char = key.char

                    # 灵巧手数字键控制（在按下时立即响应）
                    if char in '123456':
                        joint_idx = int(char) - 1
                        direction = -1 if self._is_shift_pressed else 1
                        self._hand_joint_flags[joint_idx] = True
                        self._hand_joint_direction = direction
                        logging.info(f"[DEBUG] 灵巧手关节 {joint_idx + 1} 控制, 方向: {'反向' if direction < 0 else '正向'}")

                    # o键 - 全部张开
                    elif char.lower() == 'o':
                        self._hand_open_all_flag = True
                        logging.info("[DEBUG] 灵巧手全部张开")

                    # c键 - 全部闭合（注意不是caps_lock）
                    elif char.lower() == 'c':
                        # 只有当caps_lock没有按下时才认为是c键
                        self._hand_close_all_flag = True
                        logging.info("[DEBUG] 灵巧手全部闭合")

                    # g键 - 四指同时闭合
                    elif char.lower() == 'g':
                        self._hand_grasp_four_flag = True
                        logging.info("[DEBUG] 四指同时闭合（快捷抓取）")

                    # f键 - 四指同时张开
                    elif char.lower() == 'f':
                        self._hand_release_four_flag = True
                        logging.info("[DEBUG] 四指同时张开（释放）")

                except AttributeError:
                    pass

        except AttributeError:
            pass

    def on_release(self, key):
        # 特殊键处理
        if key == keyboard.Key.ctrl_l:
            self._is_ctrl_l_pressed = False
        elif key == keyboard.Key.shift:
            self._is_shift_pressed = False
        elif key == keyboard.Key.esc:
            self._exit_flag = True
            return False
        elif key == keyboard.Key.space:
            pass  # 空格键只在on_press中处理
        elif key == keyboard.Key.alt:
            self._agent_id = 0 if self._agent_id else 1
        else:
            # 重置末端偏移
            self._end_pos_offset = np.zeros(3)
            self._end_rot_offset = np.eye(3)

            # 处理字符键
            try:
                char = key.char.lower()

                # === 其他功能键 ===
                if char == 'r':
                    self._reset_flag = True
                    logging.info("[DEBUG] 重置环境")
                elif char == 'a':
                    self._save_trajectory_flag = True
                    logging.info("[DEBUG] 保存轨迹")
                elif char == 'x':  # 用x键来清除轨迹，避免与c键冲突
                    self._clear_trajectory_flag = True
                    logging.info("[DEBUG] 清除轨迹")

            except AttributeError:
                # 重置末端偏移
                self._end_pos_offset = np.zeros(3)
                self._end_rot_offset = np.eye(3)

    def get_outputs(self):
        """获取末端偏移"""
        return (
            np.clip(self._end_pos_offset, -0.04, 0.04),
            self._end_rot_offset,
        )

    def get_hand_joint_control(self):
        """
        获取灵巧手关节控制指令

        Returns:
            tuple: (joint_controls, open_all, close_all, grasp_four, release_four)
                - joint_controls: List[tuple] [(joint_idx, direction), ...] 关索引和方向
                - open_all: bool 是否全部张开
                - close_all: bool 是否全部闭合
                - grasp_four: bool 是否四指闭合
                - release_four: bool 是否四指张开
        """
        joint_controls = []
        for i, flag in enumerate(self._hand_joint_flags):
            if flag:
                direction = getattr(self, '_hand_joint_direction', 1)
                joint_controls.append((i, direction))

        result = (
            joint_controls,
            self._hand_open_all_flag,
            self._hand_close_all_flag,
            self._hand_grasp_four_flag,
            self._hand_release_four_flag
        )

        # 重置标志
        self._hand_joint_flags = [False] * 6
        self._hand_open_all_flag = False
        self._hand_close_all_flag = False
        self._hand_grasp_four_flag = False
        self._hand_release_four_flag = False

        return result
