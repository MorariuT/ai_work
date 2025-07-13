from physics_simulator import PhysicsSimulator
from synthnova_config import (
    MujocoConfig,
    PhysicsSimulatorConfig,
    RobotConfig,
    MeshConfig,
    CuboidConfig
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
import mink
from loop_rate_limiters import RateLimiter
from auro_utils import xyzw_to_wxyz, wxyz_to_xyzw
from pathlib import Path
import numpy as np
from physics_simulator.utils.data_types import JointTrajectory
import time

from physics_simulator.utils.state_machine import SimpleStateMachine

class CubeGenerator():
    def __init__(self, no_cubes: int=10, random_state: int=42, simulation: PhysicsSimulator=None) -> None:
        self.no_cubes = no_cubes
        self.random_state = random_state
        self.sim = simulation

    def generate_cubes(
            self, 
            pos_min: int=-20, 
            pos_max: int=20, 
            colors: list=[[0, 1, 0], [1, 0, 0], [0, 0, 1]], 
            scale: list[float]=[0.05, 0.5, 0.5]) -> None:
        def generate_colors(colors: list) -> list:
            colors_ = []
            for _ in range(self.no_cubes):
                colors_.append(colors[np.random.randint(0, len(colors))])
            return colors_
        def generate_positions(min: int, max: int) -> list:
            positions = []
            for _ in range(self.no_cubes):

                pos_x = np.random.random()
                pos_y = np.random.random()
                pos_z = 1 # in default state they are on the floor

                while([pos_x, pos_y, pos_z] in positions):
                    pos_x = np.random.random()
                    pos_y = np.random.random()
                    pos_z = 1 # in default state they are on the floor

                positions.append([pos_x, pos_y, pos_z])

            return positions
        positions = generate_positions(min=pos_min, max=pos_max)
        colors = generate_colors(colors=colors)
        for _ in range(self.no_cubes):
            cube_config = CuboidConfig(
                prim_path="/World/Cube",
                position=positions[_],
                orientation=[0, 0, 0, 1],
                scale=[scale[0], scale[1], scale[2]],
                color=colors[_],
            )
            self.sim.add_object(cube_config)

        return self.sim            
