from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.buried_bornes_env import BuriedBornesEnv

# Create environment instance
env = BuriedBornesEnv()

# Check if environment follows Gym API
check_env(env, warn=True)

# Instantiate PPO 
policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=0.995,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_buried_bornes")
