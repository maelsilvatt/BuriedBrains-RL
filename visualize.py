from stable_baselines3 import PPO
from envs.buried_bornes_env import BuriedBornesEnv
import time

# Carrega o ambiente
env = BuriedBornesEnv()

# Carrega o modelo treinado
model = PPO.load("ppo_buried_bornes")

# Número de episódios de teste
num_episodes = 5

for ep in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print(f"\n=== Episódio {ep+1} ===")

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        env.render()
        print(f"Ação: {action} | Recompensa: {reward:.2f}\n")
        time.sleep(0.3)  # pequena pausa pra visualização fluida

    print(f"🏁 Episódio {ep+1} finalizado em {steps} passos | Recompensa total: {total_reward:.2f}")
