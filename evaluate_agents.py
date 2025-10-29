#!/usr/bin/env python3
"""Batch evaluation tool for trained Snake agents."""

import argparse
import csv
import json
import os
import statistics
import tempfile

MPL_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "matplotlib")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.envs.advanced_snake_env import AdvancedSnakeEnv
from src.agents.Vanilla_DQN import DQNAgent as VanillaAgent
from src.agents.Double_DQN import DQNAgent as DoubleAgent
from src.agents.Dueling_DQN import DQNAgent as DuelingAgent
from src.agents.Rainbow_DQN import DQNAgent as RainbowAgent


AGENT_CLASSES = {
    "vanilla": VanillaAgent,
    "double": DoubleAgent,
    "dueling": DuelingAgent,
    "rainbow": RainbowAgent,
}


def select_device() -> torch.device:
    """Pick the fastest locally available torch device."""
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    print(f"Using device: {device_name}")
    return torch.device(device_name)


def load_config(config_path: str) -> dict:
    """Load the shared experiment configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate_agent(agent_dir: str, env_config: dict, device: torch.device, episodes: int) -> dict:
    """
    Run a trained agent through evaluation episodes and collect summary stats.

    Each agent directory is expected to contain `agent_config.json` and `final_model.pth`.
    """
    config_path = os.path.join(agent_dir, "agent_config.json")
    model_path = os.path.join(agent_dir, "final_model.pth")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing agent config: {config_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    with open(config_path, "r") as f:
        agent_config = json.load(f)

    agent_type = agent_config["dqn_agent"]
    agent_cls = AGENT_CLASSES.get(agent_type)
    if agent_cls is None:
        raise ValueError(f"Unsupported agent type '{agent_type}' in {config_path}")

    env_kwargs = env_config.copy()
    env_kwargs["render_mode"] = False
    env = AdvancedSnakeEnv(rewards=agent_config["rewards"], **env_kwargs)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    hyperparams = agent_config.get("hyperparameters", {})
    agent = agent_cls(
        state_shape=state_shape,
        action_size=action_size,
        config=hyperparams,
        device=device,
    )
    state_dict = torch.load(model_path, map_location=device)
    agent.policy_net.load_state_dict(state_dict)
    agent.update_target_network()
    agent.policy_net.eval()

    episode_rewards: list[float] = []
    episode_scores: list[int] = []
    episode_lengths: list[int] = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

        episode_rewards.append(total_reward)
        episode_scores.append(info.get("score", 0))
        episode_lengths.append(steps)

    env.close()

    def summary(data: list[float]) -> tuple[float, float]:
        """Return mean and stdev while handling the single-episode case."""
        if len(data) == 1:
            return data[0], 0.0
        return statistics.mean(data), statistics.stdev(data)

    reward_mean, reward_std = summary(episode_rewards)
    score_mean, score_std = summary([float(s) for s in episode_scores])
    length_mean, length_std = summary([float(s) for s in episode_lengths])

    return {
        "agent_dir": os.path.basename(agent_dir),
        "agent_type": agent_type,
        "episodes": episodes,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "score_mean": score_mean,
        "score_std": score_std,
        "length_mean": length_mean,
        "length_std": length_std,
    }


def plot_results(results: list[dict], plot_path: str) -> None:
    """Create bar charts with error bars for reward, score, and episode length."""
    labels = [row["agent_dir"] for row in results]
    x = list(range(len(labels)))

    metric_specs = [
        ("Average Reward", [row["reward_mean"] for row in results], [row["reward_std"] for row in results]),
        ("Average Score", [row["score_mean"] for row in results], [row["score_std"] for row in results]),
        (
            "Average Episode Length",
            [row["length_mean"] for row in results],
            [row["length_std"] for row in results],
        ),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(3 * len(metric_specs), 4), sharey=False)
    if len(metric_specs) == 1:
        axes = [axes]

    for ax, (title, means, stds) in zip(axes, metric_specs):
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Agent")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main():
    """Parse CLI arguments and evaluate every agent under the target directory."""
    parser = argparse.ArgumentParser(description="Evaluate trained Snake agents.")
    parser.add_argument(
        "--models-root",
        help="Path with trained agent directories.",
    )
    parser.add_argument(
        "--config",
        default="config/config.json",
        help="Path to the global training configuration file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per agent.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate comparison bar charts (enabled by default).",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Write plots to this image file. Defaults to <models-root>/evaluation_results.png.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Write aggregated results to this CSV file. Defaults to <models-root>/evaluation_results.csv.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device()

    agent_paths = [
        os.path.join(args.models_root, name)
        for name in sorted(os.listdir(args.models_root))
        if os.path.isdir(os.path.join(args.models_root, name))
    ]

    if not agent_paths:
        raise RuntimeError(f"No agent directories found in {args.models_root}")

    results = []
    for agent_path in agent_paths:
        print(f"\nEvaluating {agent_path} ...")
        stats = evaluate_agent(agent_path, config["env"], device, args.episodes)
        results.append(stats)

    csv_path = args.csv_output or os.path.join(args.models_root, "evaluation_results.csv")
    fieldnames = [
        "agent_dir",
        "agent_type",
        "episodes",
        "reward_mean",
        "reward_std",
        "score_mean",
        "score_std",
        "length_mean",
        "length_std",
    ]
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    plot_path = None
    if args.plot:
        plot_path = args.plot_path or os.path.join(args.models_root, "evaluation_results.png")
        plot_results(results, plot_path)

    print("\n=== Evaluation Summary ===")
    for stats in results:
        print(
            f"{stats['agent_dir']} ({stats['agent_type']}): "
            f"reward={stats['reward_mean']:.2f}±{stats['reward_std']:.2f}, "
            f"score={stats['score_mean']:.2f}±{stats['score_std']:.2f}, "
            f"length={stats['length_mean']:.1f}±{stats['length_std']:.1f} "
            f"over {stats['episodes']} episodes"
        )
    print(f"\nResults saved to {csv_path}")
    if plot_path:
        print(f"Plots saved to {plot_path}")


if __name__ == "__main__":
    main()
