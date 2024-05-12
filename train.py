"""
Main script to train RL agents on the system environment
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from config import get_default_cfg
from system import SystemEnvironment

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

# increase font size
plt.rcParams.update({"font.size": 11})

cfg = get_default_cfg()


def train_dqn(env, total_timesteps, policy_kwargs):
    """Train a DQN agent on the given environment.

    Args:
        env (gym.Env): The environment to train on.
        total_timesteps (int): The total number of timesteps to train for.
        policy_kwargs (dict): Keyword arguments for the policy.

    Returns:
        DQN: The trained DQN agent.
    """

    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        seed=67,
    )
    model.learn(total_timesteps=total_timesteps, log_interval=1, progress_bar=True)

    return model


def evaluate_agent(env, model, num_episodes=10):
    """Evaluate the trained agent.

    Args:
        env (gym.Env): The environment to evaluate the agent on.
        model (DQN): The trained DQN agent.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        tuple: A tuple containing lists of transitions, queue_requests, and power
            consumption for each episode.
    """
    all_transitions = []
    all_queue_requests = []
    all_power = []
    all_rewards = []  # Store rewards for each episode
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        transitions = []
        queue_requests = []
        power = []
        rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Extract the integer action from the NumPy array
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            transitions.append(env.controller.current_action)
            queue_requests.append(env.queue.current_state)
            power.append(-reward)  # Reward is negative cost
            rewards.append(reward)

            ## Enhanced Logging (add this for debugging):
            # print(
            #     f"Episode: {ep}, Step: {env.time}, Action: {action}, "
            #     f"Reward: {reward}, Queue: {env.queue.current_state}, "
            #     f"Controller: {env.controller.current_state}"
            # )

        all_transitions.append(transitions)
        all_queue_requests.append(queue_requests)
        all_power.append(power)
        all_rewards.append(rewards)

    return all_transitions, all_queue_requests, all_power, all_rewards


def always_active_baseline(env, num_episodes=10):
    """Run a baseline where the controller is always active."""
    all_power = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        power = []
        while not done:
            # Always choose "go_active" action
            obs, reward, done, _, _ = env.step(0)
            power.append(-reward)
        all_power.append(power)
    return np.mean(all_power, axis=0)


def plot_results(dqn_power, baseline_power, queue_requests, episode_duration):
    """Generate result plots."""

    # FIGURE 1 - Power Consumption Comparison
    plt.figure(1)
    plt.plot(dqn_power, label="DQN", color="blue")
    plt.plot(baseline_power, label="Always Active", color="red", linestyle="dashed")

    # Calculate RMS power for both DQN and baseline
    rms_dqn = math.sqrt(sum([p**2 for p in dqn_power]) / len(dqn_power))
    rms_baseline = math.sqrt(sum([p**2 for p in baseline_power]) / len(baseline_power))

    # Draw a horizontal line at the RMS power
    plt.axhline(y=rms_dqn, color="blue", linestyle="dotted", label="RMS DQN")
    plt.axhline(
        y=rms_baseline, color="red", linestyle="dotted", label="RMS Always Active"
    )
    plt.xlim(0, episode_duration)

    plt.ylabel("Power (mW)")
    plt.xlabel("Time Step")
    plt.title("Power Consumption Comparison")
    plt.legend()

    # Add text above the line giving the value
    plt.text(
        episode_duration - 1,
        rms_dqn + 0.1,
        f"RMS DQN: {rms_dqn:.2f} mW",
        ha="right",
        va="bottom",
    )
    plt.text(
        episode_duration - 1,
        rms_baseline + 0.1,
        f"RMS Always Active: {rms_baseline:.2f} mW",
        ha="right",
        va="bottom",
    )

    # FIGURE 2 - Queue Length Over Time
    plt.figure(2)
    plt.stem(queue_requests, label="Queue Length", linefmt="r-", markerfmt="ko")
    plt.ylabel("Queue Length")
    plt.xlabel("Time Step")
    plt.title("Queue Length over Time")
    plt.legend()
    plt.xlim(0, episode_duration)

    # FIGURE 3 - Interarrivals and Power Mode
    plt.figure(3)
    plt.plot(
        env.timeline, label="Interarrivals", color="black", marker="o", linestyle="None"
    )
    go_active = [1 if i == "go_active" else 0 for i in transitions[-1]]  # last episode
    plt.plot(go_active, label="Power Mode")
    plt.yticks([0, 1], ("Sleep", "Active"))
    plt.ylabel("Power Mode / Interarrivals")
    plt.xlabel("Cycle")
    plt.title("Service Provider - Power Mode")
    plt.minorticks_on()
    plt.legend()
    plt.xlim(0, episode_duration)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the DQN agent.")
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate the saved model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="logs/dqn_model",
        help="Path to the saved model.",
    )
    args = parser.parse_args()

    env = SystemEnvironment()
    check_env(env)
    policy_kwargs = {"net_arch": [64, 64]}

    if args.eval_only:
        # Load the trained model
        model = DQN.load(args.model_path)

        # Evaluate the agent
        (
            transitions,
            queue_requests,
            all_power,
            rewards,
        ) = evaluate_agent(env, model, num_episodes=10)

        # Generate baseline
        baseline_power = always_active_baseline(env, num_episodes=10)

        dqn_power = np.mean(all_power, axis=0)

        plot_results(
            dqn_power,
            baseline_power,
            queue_requests[-1],
            cfg.num_steps,
        )

    else:
        # Train the agent
        trained_model = train_dqn(env, cfg.total_timesteps, policy_kwargs)
        trained_model.save(args.model_path)
