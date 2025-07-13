from enum import Enum
import numpy as np

class Actions(Enum):
    LEFT = 0
    RIGHT = 1

class GameDraw(Enum):
    _FLOOR = 0
    ROBOT = 1
    TARGET = 2

    def __str__(self):
        return self.name[:1]
    

class Robot:
    def __init__(self, len=10):
        self.len = len
        self.reset()
    
    def reset(self):
        self.robot_pos = [np.random.randint(0, self.len - 1)]

        self.target_pos = [np.random.randint(0, self.len - 1)]
        while(self.robot_pos == self.target_pos):
            self.target_pos = [np.random.randint(0, self.len - 1)]
        
    def perform_action(self, action:Actions) -> bool:
        if(action == 0):
            if(self.robot_pos[0] > 0): self.robot_pos[0] -= 1

        if(action == 1):
            if(self.robot_pos[0] < self.len): self.robot_pos[0] += 1

        return self.robot_pos == self.target_pos
    
    def render(self):
        for i in range(self.len):
            if(i == self.robot_pos[0]):
                print(GameDraw.ROBOT, end=' ')
            elif(i == self.target_pos[0]):
                print(GameDraw.TARGET, end=' ')
            else:
                print(GameDraw._FLOOR, end=' ')

        print()

if __name__=="__main__":

    Robot = Robot()
    while(True):
        rand_action = np.random.choice(list(Actions))
        print(rand_action)

        Robot.perform_action(rand_action)
        Robot.render()