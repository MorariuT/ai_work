
from physics_simulator import PhysicsSimulator

from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
import mink
from loop_rate_limiters import RateLimiter
from auro_utils import xyzw_to_wxyz, wxyz_to_xyzw
import time

from physics_simulator.utils.state_machine import SimpleStateMachine
from cube_generator import CubeGenerator
import matplotlib.pyplot as plt
from synthnova_config import (
    PhysicsSimulatorConfig,
    RobotConfig,
    RgbCameraConfig,
    RealsenseD435RgbSensorConfig,
    DepthCameraConfig,
    RealsenseD435DepthSensorConfig,
    MujocoConfig,
    PhysicsSimulatorConfig,
    RobotConfig,
    MeshConfig,
    CuboidConfig
)
from physics_simulator.utils import preprocess_depth
import os
import numpy as np
import cv2
from pathlib import Path

my_config = PhysicsSimulatorConfig()
physics_simulator = PhysicsSimulator(my_config)
physics_simulator.add_default_scene()

robot_path = None
front_head_depth_camera_path = None
front_head_rgb_camera_path = None

robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_charlie",
        mjcf_path=Path()
        .joinpath(physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("robot")
        .joinpath("galbot_one_charlie_description")
        .joinpath("galbot_one_charlie.xml"),
        position=[2, 0, 0],
        orientation=[0, 0, 0, 1]
    )

object_table_config = MeshConfig(
        prim_path="/World/Table",
        mjcf_path=Path()
        .joinpath(physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("default_assets")
        .joinpath("example")
        .joinpath("ioai")
        .joinpath("table")
        .joinpath("table.xml"),
        position=[0.5, 0.5, 0],
        orientation=[0, 0, 0, 0],
        scale=[0.8, 2, 0.5]
        # x, y, inaltime
    )

finish_table_config = MeshConfig(
        prim_path="/World/Table",
        mjcf_path=Path()
        .joinpath(physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("default_assets")
        .joinpath("example")
        .joinpath("ioai")
        .joinpath("table")
        .joinpath("table.xml"),
        position=[-2.65, 0, 0],
        orientation=[0, 0, 0.70711, -0.70711],
        scale=[2, 1, 0.5]
    )

tables = [object_table_config, finish_table_config]

def add_default_obj(simulator):
    global tables, robot_config, robot_path
    for table in tables: simulator.add_object(table)
    robot_path = simulator.add_robot(robot_config)
    cg = CubeGenerator(simulation=simulator, no_cubes=20)
    simulator = cg.generate_cubes(pos_min=0, pos_max=2, scale=[0.05, 0.05, 0.05])
    return simulator

def add_head_camera(simulator):
    global front_head_depth_camera_path, front_head_rgb_camera_path
    front_head_rgb_camera_config = RgbCameraConfig(
        name="front_head_rgb_camera",
        prim_path=os.path.join(
            robot_path,
            "head_link2",
            "head_end_effector_mount_link",
            "front_head_rgb_camera",
        ),
        translation=[0.09321, -0.06166, 0.033],
        rotation=[
            0.683012701855461,
            0.1830127020294028,
            0.18301270202940284,
            0.6830127018554611,
        ],
        sensor_config=RealsenseD435RgbSensorConfig(),
        parent_entity_name="galbot_one_charlie/head_end_effector_mount_link"
    )
    front_head_rgb_camera_path = simulator.add_sensor(front_head_rgb_camera_config)
    # Add front head depth camera (RealSense D435)
    front_head_depth_camera_config = DepthCameraConfig(
        name="front_head_depth_camera",
        prim_path=os.path.join(
            robot_path,
            "head_link2",
            "head_end_effector_mount_link",
            "front_head_depth_camera",
        ),
        translation=[0.09321, -0.06166, 0.033],
        rotation=[
            0.683012701855461,
            0.1830127020294028,
            0.18301270202940284,
            0.6830127018554611,
        ],
        sensor_config=RealsenseD435DepthSensorConfig(),
        parent_entity_name="galbot_one_charlie/head_end_effector_mount_link"
    )
    front_head_depth_camera_path = simulator.add_sensor(front_head_depth_camera_config)
    return simulator

def _move_joints_to_target(self, module, target_positions, steps=100):
        """Move joints from current position to target position smoothly."""
        def interpolate_joint_positions(start_positions, end_positions, steps):
            return np.linspace(start_positions, end_positions, steps).tolist()
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

def main():
    global physics_simulator, robot_config, robot_path
    physics_simulator = add_default_obj(physics_simulator)
    physics_simulator = add_head_camera(physics_simulator)

    galbot_interface_config = GalbotInterfaceConfig()

    galbot_interface_config.modules_manager.enabled_modules.append("front_head_camera")

    galbot_interface_config.robot.prim_path = robot_path

    galbot_interface_config.front_head_camera.prim_path_rgb = front_head_rgb_camera_path
    galbot_interface_config.front_head_camera.prim_path_depth = (front_head_depth_camera_path)

    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config,
        simulator=physics_simulator
    )
    galbot_interface.initialize()

    physics_simulator.play()
    physics_simulator.step(10)

    # fig, ax = plt.subplots()

    idx = 0
    while True:
        physics_simulator.step(7)
        rgb_data = galbot_interface.front_head_camera.get_rgb()
        # depth_data = galbot_interface.front_head_camera.get_depth()
        # depth_data = preprocess_depth(
        #     depth_data,
        #     scale=1000, # m to mm
        #     min_value=0.0,
        #     max_value=3 * 1000, # 3m to mm
        #     data_type=np.uint16,
        # )

        if(idx % 10 == 0): cv2.imwrite("rgb.png", rgb_data)
        idx += 1;
    


        # params = galbot_interface.front_head_camera.get_parameters()
        # intrinsic_matrix = params["rgb"]["intrinsic_matrix"]
        # print(params)
        # print("intrinsic_matrix: ", intrinsic_matrix)
    

    physics_simulator.loop()
    physics_simulator.close()

main()