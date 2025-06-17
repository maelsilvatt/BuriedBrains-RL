# 🤖 BuriedBrains-RL — A Reinforcement Learning Roguelike RPG

A custom OpenAI Gym environment for training reinforcement learning agents in a roguelike dungeon RPG world, inspired by games like *Buried Bornes*.

---

## 🎮 Overview

This project implements a grid-based dungeon environment with:

- Procedural map generation
- Turn-based combat system
- Multiple enemy types and status effects
- Agent actions: move, attack, heal, rest, flee
- Custom reward functions for learning optimal strategies

---

## 📦 Features

- Fully customizable gym environment (`BuriedBrainsEnv`)
- Plug-and-play with Stable-Baselines3 agents (DQN, PPO, A2C, etc)
- Expandable: Add loot, classes, bosses, NPCs, skills, and more
- Modular code: easy to hack and experiment

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- `gym`
- `numpy`
- `stable-baselines3`

```bash
pip install -r requirements.txt
````

### Run Training

```bash
python train_agent.py
```

---

## 🧠 RL Agents

| Agent  | Framework | Notes       |
| ------ | --------- | ----------- |
| DQN    | SB3       | Baseline    |
| PPO    | SB3       | More stable |
| Custom | PyTorch   | In progress |

---

## 📁 Structure

```
.
├── envs/                 # Gym environments
├── agents/               # RL agents
├── train_agent.py        # Entry point for training
├── visualize.py          # Game state visualizer
├── README.md
```

---

## ✨ Goals

* 🧠 Train a competent agent to survive deep dungeon levels
* 🔬 Explore curriculum learning, transfer learning
* 🛡️ Add RPG mechanics: loot, inventory, skill trees
* 💥 Make it fun to watch!

---

## 🤝 Contribute

Pull requests, feedback and feature ideas are welcome!

---

## 📜 License

GNU 3.0

```
