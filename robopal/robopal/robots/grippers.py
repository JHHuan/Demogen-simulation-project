import numpy as np

REGISTERED_ENDS = {}


class EndMetaClass(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)

        if not cls.__name__ == "BaseEnd":
            REGISTERED_ENDS[cls.__name__] = cls
        return cls


class BaseEnd(object, metaclass=EndMetaClass):

    gripper_joint_names = dict()
    gripper_joint_indexes = dict()
    gripper_actuator_names = dict()
    gripper_actuator_indexes = dict()

    # dt will pass in after the environment is instantiated
    dt: int = None

    _ctrl_range = [-1, 1]

    def __init__(self, robot_data, robot_model, agent) -> None:

        self.robot_data = robot_data
        self.robot_model = robot_model
        self.agent_id = agent[-1]

    def apply_action(self, action: np.ndarray) -> None:
        pass

    def open(self):
        self.apply_action(self._ctrl_range[1])

    def close(self):
        self.apply_action(self._ctrl_range[0])

    def get_finger_observations(self):
        pass

    def reset(self):
        pass


class RethinkGripper(BaseEnd):

    _ctrl_range = [-0.01, 0.02]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_gripper_l_finger_joint').ctrl[0] = action
        self.robot_data.actuator(f'{self.agent_id}_gripper_r_finger_joint').ctrl[0] = action
    
    def get_finger_observations(self):
        return np.concatenate([
            self.robot_data.joint(f'{self.agent_id}_l_finger_joint').qpos,
            self.robot_data.joint(f'{self.agent_id}_l_finger_joint').qvel * self.dt
        ], axis=0)


class RobotiqGripper(BaseEnd):

    _ctrl_range = [0, 0.83]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_robotiq_2f_85').ctrl[0] = action

    def open(self):
        self.apply_action(self._ctrl_range[0])

    def close(self):
        self.apply_action(self._ctrl_range[1])


class PandaHand(BaseEnd):

    _ctrl_range = [0, 255]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_actuator8').ctrl[0] = action

    def get_finger_observations(self):
        return np.concatenate([
            self.robot_data.joint(f'{self.agent_id}_finger_joint1').qpos,
            self.robot_data.joint(f'{self.agent_id}_finger_joint1').qvel * self.dt
        ], axis=0)


class O6Hand(BaseEnd):
    """O6 dexterous hand with 6 DOF (2 for thumb, 1 for each of 4 fingers).

    The distal joints are coupled to proximal joints via equality constraints.
    - Thumb: 2 DOF (yaw + pitch, IP joint coupled)
    - Index/Middle/Ring/Pinky: 1 DOF each (MCP joint, DIP coupled)
    """

    # 6 degrees of freedom
    _ctrl_range = [-100, 100]

    # Control joint names (without agent prefix)
    CONTROL_JOINTS = [
        'thumb_cmc_yaw',      # thumb abduction/adduction
        'thumb_cmc_pitch',    # thumb flexion (IP joint coupled)
        'index_mcp_pitch',    # index flexion (DIP coupled)
        'middle_mcp_pitch',   # middle flexion (DIP coupled)
        'ring_mcp_pitch',     # ring flexion (DIP coupled)
        'pinky_mcp_pitch'     # pinky flexion (DIP coupled)
    ]

    # Actuator names (without agent prefix)
    ACTUATOR_NAMES = [
        'actuator_thumb_cmc_yaw',
        'actuator_thumb_cmc_pitch',
        'actuator_index_mcp',
        'actuator_middle_mcp',
        'actuator_ring_mcp',
        'actuator_pinky_mcp'
    ]

    def __init__(self, robot_data, robot_model, agent):
        super().__init__(robot_data, robot_model, agent)
        self.n_dof = 6

    def _get_prefixed_name(self, name):
        """Add agent prefix to name."""
        return f'{self.agent_id}_{name}'

    def apply_action(self, action: np.ndarray) -> None:
        """
        Apply action to 6 finger joints.

        Args:
            action: np.ndarray of shape (6,) in order:
                    [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
                    Values in radians or joint positions
        """
        action = np.array(action)
        assert action.shape[0] == self.n_dof, \
            f"O6Hand requires {self.n_dof} actions, got {action.shape[0]}"

        for i, actuator_name in enumerate(self.ACTUATOR_NAMES):
            prefixed_name = self._get_prefixed_name(actuator_name)
            self.robot_data.actuator(prefixed_name).ctrl[0] = action[i]

    def open(self):
        """Open all fingers (0 = open)."""
        open_action = np.zeros(self.n_dof)
        self.apply_action(open_action)

    def close(self):
        """Close all fingers to max flexion."""
        close_action = np.array([
            1.54,    # thumb yaw (max abduction)
            0.52,    # thumb pitch (max flexion)
            1.57,    # index (max flexion)
            1.57,    # middle
            1.57,    # ring
            1.57     # pinky
        ])
        self.apply_action(close_action)

    def get_finger_observations(self):
        """
        Get observations for controlled finger joints.

        Returns:
            np.ndarray of shape (12,) containing positions and velocities
            for 6 controlled joints: [pos1, vel1, pos2, vel2, ...]
        """
        observations = []
        for joint_name in self.CONTROL_JOINTS:
            prefixed_name = self._get_prefixed_name(joint_name)
            joint = self.robot_data.joint(prefixed_name)
            observations.append(joint.qpos[0])
            observations.append(joint.qvel[0] * self.dt if self.dt is not None else 0)
        return np.array(observations)

    def get_fingertip_positions(self):
        """
        Get positions of all 5 fingertips.

        Returns:
            dict mapping finger names to tip positions (np.ndarray)
        """
        tip_names = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        positions = {}
        for tip_name in tip_names:
            try:
                prefixed_name = self._get_prefixed_name(tip_name)
                site = self.robot_data.site(prefixed_name)
                positions[tip_name] = site.xpos.copy()
            except:
                positions[tip_name] = np.zeros(3)
        return positions

    def reset(self):
        """Reset all joints to open position."""
        neutral_action = np.zeros(self.n_dof)
        self.apply_action(neutral_action)
