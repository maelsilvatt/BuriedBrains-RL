from stable_baselines3 import PPO
from envs.buried_bornes_env import BuriedBornesEnv
import time

# Load the environment
env = BuriedBornesEnv()

# Load the trained model
model = PPO.load("ppo_buried_bornes")

num_episodes = 5

for ep in range(num_episodes):    
    obs, info = env.reset()
        
    total_reward = 0
    steps = 0

    print(f"\n=== Episode {ep+1} ===")
    
    while True:       
        action, _ = model.predict(obs, deterministic=True)
                
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1

        env.render()
        print(f"Action: {action} | Reward: {reward:.2f}\n")
        time.sleep(0.3)
        
        if terminated or truncated:
            break

    print(f"🏁 Episode {ep+1} finished in {steps} steps | Total Reward: {total_reward:.2f}")