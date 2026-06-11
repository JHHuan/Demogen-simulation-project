import os

import numpy as np
from robopal.robots.base import BaseRobot

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class Panda(BaseRobot):
    """ Panda robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='Panda',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_attachment',
        )
        self.arm_joint_names = {self.agents[0]: ['0_joint1', '0_joint2', '0_joint3', '0_joint4', '0_joint5', '0_joint6', '0_joint7']}
        self.arm_actuator_names = {self.agents[0]: ['0_actuator1', '0_actuator2', '0_actuator3', '0_actuator4', '0_actuator5', '0_actuator6', '0_actuator7']}
        self.base_link_name = {self.agents[0]: '0_link0'}
        self.end_name = {self.agents[0]: '0_attachment'}

        self.pos_max_bound = np.array([0.6, 0.2, 0.37])
        self.pos_min_bound = np.array([0.3, -0.2, 0.02])

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.61,  -0.84,  0.47, -2.54,  0.35,  1.75, 0.44])}


class PandaGrasp(Panda):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='PandaHand',
                         mount='top_point')

        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')
        
    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.61,  -0.84,  0.47, -2.54,  0.35,  1.75, 0.44])}
    

class PandaPickAndPlace(PandaGrasp):

    def add_assets(self):
        super().add_assets()
        goal_site = """<site name="goal_site" pos="0.4 0.0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)


class PandaTripleStack(PandaGrasp):

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/red_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'red_block', {'pos': '0.5 -0.1 0.46'})

        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'green_block', {'pos': '0.5 0.0 0.46'})

        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/blue_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'blue_block', {'pos': '0.5 0.1 0.46'})

        r_goal_site = """<site name="red_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', r_goal_site)

        g_goal_site = """<site name="green_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="0 1 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', g_goal_site)

        b_goal_site = """<site name="blue_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="0 0 1 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', b_goal_site)


class PandaDemoGen(Panda):
    """Panda robot with DemoGen camera configuration for point cloud collection."""

    def __init__(self):
        super().__init__(
            scene='grasping_demogen',  # 使用带 DemoGen 相机的场景
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        # 添加可操作的物体
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')

        # 添加红色立方体
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/red_cube/body.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaSingleCube(Panda):
    """Panda robot with single cube for pick-and-place task."""

    def __init__(self):
        super().__init__(
            scene='pick_single_cube',  # 使用新的单立方体场景
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        # 只添加绿色立方体（单物体抓取任务）
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaWithO6Hand(Panda):
    """Panda robot equipped with O6 dexterous hand (6 DOF)."""

    def __init__(self):
        super().__init__(
            scene='grasping_demogen',
            gripper='O6Hand',
            mount='top_point'
        )
        # O6Hand的末端执行器名称
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        """Add objects for manipulation tasks."""
        # 添加可操作的物体
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/metaworld_box/metaworld_box.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}

class PandaSingleCube1(Panda):
    """Panda robot with single cube for pick-and-place task."""

    def __init__(self):
        super().__init__(
            scene='pick_single_cube_1',  # 使用新的单立方体场景
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        # 只添加绿色立方体（单物体抓取任务）
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube_1.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaSingleCube2cam(Panda):
    """Panda robot with single cube, dual camera (front-left/right 45°)."""

    def __init__(self):
        super().__init__(
            scene='pick_single_cube_2cam',
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube_1.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaSingleCube3cam(Panda):
    """Panda robot with single cube, three cameras (front + left/right 45°)."""

    def __init__(self):
        super().__init__(
            scene='pick_single_cube_3cam',
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube_1.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaButtonBox3cam(Panda):
    """Panda robot with buttonbox (press button task), three cameras."""

    def __init__(self):
        super().__init__(
            scene='press_button_3cam',
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/buttonbox/buttonbox.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}


class PandaAssembly3cam(Panda):
    """Panda robot with assembly objects (round nut + peg), three cameras."""

    def __init__(self):
        super().__init__(
            scene='assembly_3cam',
            gripper='PandaHand',
            mount='top_point'
        )
        self.end_name = {self.agents[0]: '0_eef'}

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/assembly/assembly.xml')

    @property
    def init_qpos(self):
        """Robot's init joint position."""
        return {self.agents[0]: np.array([-0.61, -0.84, 0.47, -2.54, 0.35, 1.75, 0.44])}
