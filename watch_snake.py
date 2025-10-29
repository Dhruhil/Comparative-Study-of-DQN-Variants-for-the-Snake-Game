#!/usr/bin/env python3

import torch
import cv2
from src.envs.advanced_snake_env import AdvancedSnakeEnv
import sys
import os
import json

def load_config(path="./config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

def select_device():
    if torch.cuda.is_available():
        dev_name = "cuda"
    elif torch.mps.is_available():
        dev_name = "mps"
    else:
        dev_name = "cpu"
    print(f"Device: {dev_name}")
    return torch.device(dev_name)

def play_trained_model(save_path, speed=0.002):
    with open(os.path.join(save_path, "agent_config.json"), "r") as f:
        agent_config = json.load(f)

    model_path = os.path.join(save_path, "final_model.pth")
    env = AdvancedSnakeEnv(rewards=agent_config["rewards"], **config["env"])
    device = select_device()
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    match agent_config["dqn_agent"]:
        case "vanilla":
            from src.agents.Vanilla_DQN import load_trained_agent
        case "double":
            from src.agents.Double_DQN import load_trained_agent
        case "dueling":
            from src.agents.Dueling_DQN import load_trained_agent
        case "rainbow":
            from src.agents.Rainbow_DQN import load_trained_agent
        case _:
            raise NotImplementedError(f"Unexpected name: {agent_config["dqn_agent"]}")

    agent = load_trained_agent(model_path, state_shape, action_size, device)
    print(f">>> Loaded trained model: {model_path} <<<")

    for ep in range(3):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            img = env.render(mode="rgb_array")
            if img is not None:
                cv2.imshow("Trained Snake Demo", img)
                if cv2.waitKey(int(speed * 1000)) & 0xFF == ord('q'):
                    done = True
                    break

        print(f"Episode {ep+1}: Reward={total_reward:.2f}, Score={info.get('score', 0)}")

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    save_path = sys.argv[1]
    play_trained_model(save_path)

