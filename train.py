#!/usr/bin/env python3

import json
import torch
import random
import numpy as np
import os
import sys
import cv2
from datetime import datetime
import subprocess
from argparse import ArgumentParser

from src.envs.advanced_snake_env import AdvancedSnakeEnv
from src.agents.Vanilla_DQN import DQNAgent as VanillaDQNAgent
from src.agents.Double_DQN import DQNAgent as DoubleDQNAgent
from src.agents.Dueling_DQN import DQNAgent as DuelingDQNAgent
from src.agents.Rainbow_DQN import DQNAgent as RainbowDQNAgent
from src.training.trainer import Trainer

def select_device():
    if torch.cuda.is_available():
        dev_name = "cuda"
    elif torch.mps.is_available():
        dev_name = "mps"
    else:
        dev_name = "cpu"
    print(f"Device: {dev_name}")
    return torch.device(dev_name)

agents = {
    "vanilla": VanillaDQNAgent,
    "double": DoubleDQNAgent,
    "dueling": DuelingDQNAgent,
    "rainbow": RainbowDQNAgent,
}

# -----------------------------
# Utility functions
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path="./config/config.json"):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Play / Visual Demo
# -----------------------------
def play_agent(env, agent, episodes=2, speed=0.1):
    print("\n🎮 Evaluating trained DQN agent...")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            img = env.render(mode='rgb_array')
            if img is not None:
                cv2.imshow("DQN Snake", img)
                if cv2.waitKey(int(speed * 1000)) & 0xFF == ord('q'):
                    break

        print(f"Episode {ep + 1}: Reward={total_reward:.2f}, Score={info.get('score', 0)}")
    cv2.destroyAllWindows()

def do_train(agent_config, default_hyperparams, default_rewards, config, index, save_dir_root, device):
    agent_name = agent_config["dqn_agent"]
    hyperparams = default_hyperparams | agent_config["hyperparameters"]
    rewards = default_rewards | agent_config["rewards"]
    # Initialize environment
    env = AdvancedSnakeEnv(rewards=rewards, **config["env"])

    name = f"{index}-{agent_name}"

    save_dir = os.path.join(save_dir_root, f"{index:02d}-{agent_name}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "agent_config.json"), "w") as f:
        json.dump({
            "dqn_agent": agent_name,
            "hyperparameters": hyperparams,
            "rewards": rewards
        }, f, indent=4)

    DQNAgent = agents[agent_name]
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        config=hyperparams,
        device=device
    )

    trainer_config = {
        "episodes": hyperparams["episodes"],
        "target_update_interval": hyperparams["target_update_freq"],
        "save_path": save_dir
    }

    trainer = Trainer(name, agent, env, trainer_config)
    trainer.train()

    model_path = os.path.join(save_dir, "final_model.pth")
    agent.save(model_path)
    print(f">>> Final model saved at {model_path} <<<")

    env.close()

# -----------------------------
# Main Training Logic
# -----------------------------
def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--index", "-i", help="Index of the agent", type=int, dest="index")
    arg_parser.add_argument("--path-prefix", "-p", help="Path prefix", type=str, dest="prefix")
    args = arg_parser.parse_args()
    arg_index = args.index
    save_dir_root = args.prefix

    config = load_config()
    set_seed(config.get("seed", 42))

    device = select_device()
    print(f"Device: {device}")    

    if not save_dir_root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir_root = os.path.join(config["train"]["output_dir"], f"snake_{timestamp}")

    default_hyperparams = config["defaults"]["hyperparameters"]
    default_rewards = config["defaults"]["rewards"]
    if arg_index is None:
        processes: list[subprocess.Popen] = []
        for index in range(len(config["agents"])):
            processes.append(subprocess.Popen([sys.executable] + sys.argv + ["-i", str(index), "-p", save_dir_root]))
        for p in processes:
            p.wait()
    else:
        agent_config = config["agents"][arg_index]
        do_train(agent_config, default_hyperparams, default_rewards, config, arg_index, save_dir_root, device)


if __name__ == "__main__":
    main()
