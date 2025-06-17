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
        
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False 
        truncated = False

        self.last_pos = list(self.agent_pos)

        # Movement actions (0-3)
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
        
        # Attack action (4) - now independent of movement
        if action == 4:
            if np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos)) <= 1.0:  # Attack in adjacent cells
                self.enemy_hp -= self.agent_damage
                reward += 1.0  # Smaller reward for hitting
                if self.enemy_hp <= 0:
                    reward += 10  # Big reward for killing
            else:
                reward -= 0.2  # Small penalty for attacking air

        # Distance-based reward
        dist_new = np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos))
        reward += (dist_old - dist_new) * 0.1  # Reduced distance factor

        # Enemy turn if still alive
        if self.enemy_hp > 0:
            # Enemy movement towards agent
            if self.enemy_pos != self.agent_pos:
                dx = self.agent_pos[0] - self.enemy_pos[0]
                dy = self.agent_pos[1] - self.enemy_pos[1]

                if abs(dx) > abs(dy):
                    self.enemy_pos[0] += 1 if dx > 0 else -1
                else:
                    self.enemy_pos[1] += 1 if dy > 0 else -1

            # Enemy attack if adjacent
            if np.linalg.norm(np.array(self.agent_pos) - np.array(self.enemy_pos)) <= 1.0:
                self.agent_hp -= self.enemy_damage
                reward -= 2.0  # Increased penalty for being hit

        # Check termination conditions
        if self.enemy_hp <= 0:
            reward += 20  # Final reward for victory
            terminated = True
        if self.agent_hp <= 0:
            reward -= 10  # Final penalty for death
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