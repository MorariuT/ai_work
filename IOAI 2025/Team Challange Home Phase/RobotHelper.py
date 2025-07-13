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

def add_robot_to_scene()