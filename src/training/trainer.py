import torch
import matplotlib.pyplot as plt
import os
import numpy as np


class Trainer:
    def __init__(self, name, agent, env, config):
        self.name = name
        self.agent = agent
        self.env = env
        self.config = config
        self.episodes = config.get("episodes", 500)
        self.target_update = config.get("target_update_interval", 100)
        self.save_path = config.get("save_path", "models/")
        self.rewards = []
        self.losses = []

        os.makedirs(self.save_path, exist_ok=True)

    def train(self):
        best_avg_reward = float("-inf")
        recent_rewards = []

        csv_log = ["Episode,Epsilon,Reward,Avg10,Loss\n"]

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward, total_loss = 0, 0
            done = False

            while not done:
                action = self.agent.act(state)
                try:
                    epsilon = self.agent.epsilon
                except AttributeError:
                    epsilon = None
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.replay()
                if loss:
                    total_loss += loss

                state = next_state
                total_reward += reward

            self.rewards.append(total_reward)
            self.losses.append(total_loss)

            recent_rewards.append(total_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            avg_reward = np.mean(recent_rewards)

            if epsilon is not None:
                print(
                    f"{self.name} Episode {episode+1}/{self.episodes} | Epsilon: {epsilon:.3f} | "
                    f"Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f} | "
                    f"Loss: {total_loss:.4f}"
                )
            else:
                print(
                    f"Episode {episode+1}/{self.episodes} | "
                    f"Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f} | "
                    f"Loss: {total_loss:.4f}"
                )
            csv_log.append(f"{episode+1},{epsilon},{total_reward},{avg_reward},{total_loss}\n")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = os.path.join(self.save_path, "best_model.pth")
                self.agent.save(best_path)
                print(f">>> New best model saved! Avg Reward = {best_avg_reward:.2f} <<<")

            if (episode + 1) % self.target_update == 0:
                self.agent.update_target_network()

            if episode > 0 and episode % 100 == 0:
                checkpoint_path = os.path.join(self.save_path, f"checkpoint_{episode}.pth")
                self.agent.save(checkpoint_path)

        final_path = os.path.join(self.save_path, "final_model.pth")
        final_csv_path = os.path.join(self.save_path, "training-log.csv")
        self.agent.save(final_path)
        print(f">>> Final model saved at {final_path} <<<")
        with open(final_csv_path, "w") as csv_file:
            csv_file.write("".join(csv_log))

        self._plot_training_curves()

    def _plot_training_curves(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(self.rewards, label="Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.plot(self.losses, label="Loss")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "training_curves.png")
        plt.savefig(plot_path)
        plt.close()

        print(f">>> Training curves saved at {plot_path} <<<")
