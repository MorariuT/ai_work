# file: task_simulation.py

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MeshConfig
from pathlib import Path

def build_env():
    sim = PhysicsSimulator(PhysicsSimulatorConfig())
    sim.add_default_scene()

    # Add robot
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot",
        mjcf_path=Path(sim.synthnova_assets_directory) / "synthnova_assets/robot/galbot_one_charlie_description/galbot_one_charlie.xml",
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    sim.add_robot(robot_config)

    # Add shelf
    shelf = MeshConfig(
        name="shelf",
        prim_path="/World/Shelf",
        mjcf_path=Path(sim.synthnova_assets_directory) / "synthnova_assets/default/shelves/1/model/mjcf/convex_decomposition.xml",
        position=[1.0, 0.0, 0],
        orientation=[0, 0, 0, 1]
    )
    sim.add_object(shelf)

    # Add item (bottle)
    bottle = MeshConfig(
        name="bottle",
        prim_path="/World/Bottle",
        mjcf_path=Path(sim.synthnova_assets_directory) / "synthnova_assets/default/skus/1/model/mjcf/convex_decomposition.xml",
        position=[0.5, 0.0, 0.1],
        orientation=[0, 0, 0, 1],
        scale=[0.122, 0.122, 0.122]
    )
    sim.add_object(bottle)

    sim.initialize()
    return sim

if __name__ == "__main__":
    sim = build_env()
    sim.loop()
    sim.close()
