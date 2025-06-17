import gymnasium as gym
import numpy as np
from gymnasium import spaces 

class BuriedBornesEnv(gym.Env):
    """
    Custom RL Environment inspired by Buried Bornes and roguelike RPG mechanics.
    The agent navigates a grid-based dungeon, engages in combat, and aims to survive and explore.
    """

    def __init__(self, grid_size=10):
        super(BuriedBornesEnv, self).__init__()
        self.grid_size = grid_size
        self.max_hp = 10
        self.enemy_damage = 2
        self.agent_damage = 3

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_size, self.max_hp),
            shape=(6,),
            dtype=np.int32
        )    

    def reset(self, seed=None, options=None):        
        super().reset(seed=seed)

        self.agent_pos = [0, 0]
        self.agent_hp = self.max_hp

        self.enemy_pos = self.np_random.integers(0, self.grid_size, size=2).tolist()
        while self.enemy_pos == self.agent_pos:
            self.enemy_pos = self.np_random.integers(0, self.grid_size, size=2).tolist()
        self.enemy_hp = self.max_hp
        
        return self._get_obs(), {}

    def step(self, action):
        reward = -0.1        
        terminated = False 
        truncated = False  

        # Movement actions
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        # Attack action
        elif action == 4:
            if self.agent_pos == self.enemy_pos and self.enemy_hp > 0:
                self.enemy_hp -= self.agent_damage
                reward += 1
        
        # Enemy turn if still alive
        if self.enemy_hp > 0:
            if self.agent_pos == self.enemy_pos:
                self.agent_hp -= self.enemy_damage
                reward -= 1

        # Check end conditions
        if self.enemy_hp <= 0:
            reward += 5
            terminated = True
        if self.agent_hp <= 0:
            reward -= 5
            terminated = True
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1], self.agent_hp,
            self.enemy_pos[0], self.enemy_pos[1], self.enemy_hp
        ], dtype=np.int32)

    def render(self, mode='human'):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.agent_pos
        grid[y][x] = 'A'
        if self.enemy_hp > 0:
            ex, ey = self.enemy_pos
            grid[ey][ex] = 'E'
        for row in grid:
            print(' '.join(row))
        print(f"Agent HP: {self.agent_hp} | Enemy HP: {self.enemy_hp}\n")