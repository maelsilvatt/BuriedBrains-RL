from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.buried_bornes_env import BuriedBornesEnv

# Create environment instance
env = BuriedBornesEnv()

# Check if environment follows Gym API
check_env(env, warn=True)

# Instantiate PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_buried_bornes")

# Load and test
model = PPO.load("ppo_buried_bornes")
obs = env.reset()

for _ in range(20):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
