import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import robot_define as rd
import numpy as np

register(
    id='chacha-robot',                       
    entry_point='robot_env:Robot_Env'
)

class Robot_Env(gym.Env):

    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, lenght=10, render_mode=None):
        self.lenght = 10

        self.Robot = rd.Robot(len=self.lenght)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(rd.Actions))

        self.observation_space = spaces.Box(
            low=0,
            high=self.lenght - 1,
            shape=(2,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.Robot.reset()

        info = {}

        obs = np.concatenate((self.Robot.robot_pos, self.Robot.target_pos))

        self.Robot.render()

        return obs, info
    
    def step(self, action):
        self.Robot.perform_action(action=action)

        reward = 0
        terminated = False
        if(self.Robot.robot_pos == self.Robot.target_pos):
            reward = 1
            terminated = True

        obs = np.concatenate((self.Robot.robot_pos, self.Robot.target_pos))

        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, reward, terminated, False, info
    
    def render(self):
        self.Robot.render()


if __name__=="__main__":
    env = gym.make('chacha-robot', render_mode='human')

    obs = env.reset()[0]

    while(True):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]