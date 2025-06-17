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

print("\n--- INICIANDO TESTE DO MODELO TREINADO ---\n")

# Load and test
model = PPO.load("ppo_buried_bornes")

obs, info = env.reset()

for i in range(20):    
    while True:
        action, _states = model.predict(obs, deterministic=True)
                
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            print(f"Episódio {i+1} finalizado.")            
            obs, info = env.reset()
            break 