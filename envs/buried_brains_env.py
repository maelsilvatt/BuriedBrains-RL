import gymnasium as gym
import numpy as np
from gymnasium import spaces 

class BuriedBrainsEnv(gym.Env):
    """
    Custom RL Environment inspired by Buried Bornes and roguelike RPG mechanics.
    The agent navigates a grid-based dungeon, engages in combat, and aims to survive and explore.
    """

    def __init__(self, grid_size=10):
        super(BuriedBrainsEnv, self).__init__()
        self.grid_size = grid_size
        self.max_hp = 10
        self.enemy_damage = 2
        self.agent_damage = 3
        self.last_pos = None

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-self.grid_size,
            high=self.grid_size,
            shape=(6,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = [0, 0]
        self.agent_hp = self.max_hp

        min_dist = self.grid_size // 2
        while True:
            self.enemy_pos = self.np_random.integers(0, self.grid_size, size=2).tolist()
            dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos))
            if dist > min_dist:
                break
        
        self.enemy_hp = self.max_hp
        self.last_pos = self.agent_pos
        
        return self._get_obs(), {}

    def step(self, action):
        dist_old = np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos))
        
        reward = -0.01
        terminated = False 
        truncated = False

        self.last_pos = list(self.agent_pos)

        # Movement actions
        moved = False
        if action == 0:  # Up
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
                moved = True
        elif action == 1:  # Down
            if self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
                moved = True
        elif action == 2:  # Left
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
                moved = True
        elif action == 3:  # Right
            if self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
                moved = True

        if action in [0, 1, 2, 3] and not moved:
            reward -= 0.5
        
        if self.agent_pos == self.last_pos:
            reward -= 0.5

        # Attack action
        elif action == 4:
            if self.agent_pos == self.enemy_pos and self.enemy_hp > 0:
                self.enemy_hp -= self.agent_damage
                reward += 10

        dist_new = np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos))
        distance_factor = 0.5
        reward_dist = (dist_old - dist_new) * distance_factor
        reward += reward_dist
        
        # Enemy turn if still alive
        if self.enemy_hp > 0:
            if self.enemy_pos != self.agent_pos:
                dx = self.agent_pos[0] - self.enemy_pos[0]
                dy = self.agent_pos[1] - self.enemy_pos[1]

                if abs(dx) > abs(dy):
                    if dx > 0:
                        self.enemy_pos[0] += 1
                    else:
                        self.enemy_pos[0] -= 1
                else:
                    if dy > 0:
                        self.enemy_pos[1] += 1
                    else:
                        self.enemy_pos[1] -= 1

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
        relative_pos = np.array(self.enemy_pos) - np.array(self.agent_pos)
        
        return np.array([
            self.agent_pos[0], self.agent_pos[1], self.agent_hp,
            relative_pos[0],
            relative_pos[1],
            self.enemy_hp
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