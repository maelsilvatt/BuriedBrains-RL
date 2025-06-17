import arcade
from stable_baselines3 import PPO
from envs.buried_brains_env import BuriedBrainsEnv

# --- Window Constants ---
TILE_SIZE = 60
GRID_WIDTH = 10
GRID_HEIGHT = 10
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE
SCREEN_TITLE = "Buried Brains - Agent in Action"

# --- Colors ---
COLOR_BACKGROUND = arcade.color.BLACK
COLOR_GRID = arcade.color.GRAY
COLOR_AGENT = arcade.color.BLUE
COLOR_ENEMY = arcade.color.RED
COLOR_HP_BACKGROUND = arcade.color.DARK_RED
COLOR_HP_FOREGROUND = arcade.color.GREEN_YELLOW


class GameVisualizer(arcade.Window):
    """
    Main window to visualize the agent's environment.
    This class manages the rendering and game update logic.
    """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(COLOR_BACKGROUND)

        # Load environment and model
        self.env = BuriedBrainsEnv()
        self.model = PPO.load("ppo_buried_bornes")
        
        # Reset environment to initial state
        self.obs, self.info = self.env.reset()
        
        self.total_reward = 0
        self.steps = 0
        self.episode_count = 1

    def grid_to_pixels(self, grid_x, grid_y):
        """ Convert grid coordinates to screen pixels. """
        pixel_x = grid_x * TILE_SIZE + TILE_SIZE / 2
        # Invert Y because the environment grid has (0,0) at the top and Arcade at the bottom
        pixel_y = (self.env.grid_size - 1 - grid_y) * TILE_SIZE + TILE_SIZE / 2
        return pixel_x, pixel_y

    def on_draw(self):
        """ All drawing magic happens here. """
        self.clear()
        
        # Draw grid
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            arcade.draw_line(x, 0, x, SCREEN_HEIGHT, COLOR_GRID, 1)
        for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
            arcade.draw_line(0, y, SCREEN_WIDTH, y, COLOR_GRID, 1)

        # Draw agent
        agent_px, agent_py = self.grid_to_pixels(self.env.agent_pos[0], self.env.agent_pos[1])
        arcade.draw_rectangle_filled(agent_px, agent_py, TILE_SIZE, TILE_SIZE, COLOR_AGENT)

        # Draw enemy if alive
        if self.env.enemy_hp > 0:
            enemy_px, enemy_py = self.grid_to_pixels(self.env.enemy_pos[0], self.env.enemy_pos[1])
            arcade.draw_rectangle_filled(enemy_px, enemy_py, TILE_SIZE, TILE_SIZE, COLOR_ENEMY)

        # --- Draw UI ---
        # Agent HP
        arcade.draw_text("Agent HP:", 10, SCREEN_HEIGHT - 30, arcade.color.WHITE, 16)
        arcade.draw_rectangle_filled(130, SCREEN_HEIGHT - 22, 150, 20, COLOR_HP_BACKGROUND)
        agent_hp_width = max(0, (self.env.agent_hp / self.env.max_hp) * 150)
        arcade.draw_rectangle_filled(130, SCREEN_HEIGHT - 22, agent_hp_width, 20, COLOR_HP_FOREGROUND)

        # Enemy HP
        arcade.draw_text("Enemy HP:", SCREEN_WIDTH - 220, SCREEN_HEIGHT - 30, arcade.color.WHITE, 16)
        arcade.draw_rectangle_filled(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 22, 150, 20, COLOR_HP_BACKGROUND)
        enemy_hp_width = max(0, (self.env.enemy_hp / self.env.max_hp) * 150)
        arcade.draw_rectangle_filled(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 22, enemy_hp_width, 20, COLOR_HP_FOREGROUND)
        
        # Episode info
        arcade.draw_text(f"Episode: {self.episode_count}", 10, 10, arcade.color.WHITE, 14)
        arcade.draw_text(f"Steps: {self.steps}", 150, 10, arcade.color.WHITE, 14)
        arcade.draw_text(f"Total Reward: {self.total_reward:.2f}", 280, 10, arcade.color.WHITE, 14)

    def on_update(self, delta_time):
        """ Game logic is executed here, every frame. """
        action, _ = self.model.predict(self.obs, deterministic=True)
        
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        
        self.total_reward += reward
        self.steps += 1

        if terminated or truncated:
            print(f"🏁 Episode {self.episode_count} finished in {self.steps} steps | Total Reward: {self.total_reward:.2f}")
            self.obs, self.info = self.env.reset()
            self.episode_count += 1
            self.total_reward = 0
            self.steps = 0


def main():
    """ Main function to run the game. """
    window = GameVisualizer(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()


if __name__ == "__main__":
    main()