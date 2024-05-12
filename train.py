"""
Main script to train RL agents on the system environment
"""

import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from config import get_default_cfg
from system import SystemEnvironment

plt.rcParams["font.family"] = "monospace"
# plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["figure.figsize"] = (7, 5)  # set default size of plots


# increase font size
plt.rcParams.update({"font.size": 10})

cfg = get_default_cfg()


def train_agent(env, total_timesteps, policy_kwargs, agent_type="DQN"):
    """Train an RL agent on the given environment.

    Args:
        env (gym.Env): The environment to train on.
        total_timesteps (int): The total number of timesteps to train for.
        policy_kwargs (dict): Keyword arguments for the policy.
        agent_type (str): The type of RL agent to train ('DQN' or 'PPO').

    Returns:
        DQN or PPO: The trained RL agent.
    """

    if agent_type == "DQN":
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            policy_kwargs=policy_kwargs,
            learning_rate=2.3e-4,
            batch_size=256,
            buffer_size=10000,
            train_freq=4,
            gamma=0.9,
            seed=42,
        )
    elif agent_type == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            policy_kwargs=policy_kwargs,
            learning_rate=7.5e-4,
            n_steps=200,
            batch_size=200,
            gamma=0.9,
            seed=42,
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    model.learn(total_timesteps=total_timesteps, log_interval=1, progress_bar=True)

    return model


def evaluate_agent(env, model, num_episodes=10):
    """Evaluate the trained agent.

    Args:
        env (gym.Env): The environment to evaluate the agent on.
        model (DQN or PPO): The trained RL agent.
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


def random_action_baseline(env, num_episodes=10):
    """Run a baseline where the controller takes random actions."""
    all_power = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        power = []
        while not done:
            # Choose random action
            action = random.randint(0, 1)
            obs, reward, done, _, _ = env.step(action)
            power.append(-reward)
        all_power.append(power)
    return np.mean(all_power, axis=0)


def threshold_based_sleep_baseline(
    env, sleep_threshold=2, wake_threshold=6, num_episodes=10
):
    """Run a baseline where the controller sleeps based on queue thresholds."""
    all_power = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        power = []
        while not done:
            # Choose action based on threshold
            if env.queue.current_state <= sleep_threshold:
                action = 1  # go to sleep
            elif env.queue.current_state >= wake_threshold:
                action = 0  # wake up
            else:
                action = 0  # stay in current state (active if already active)
            obs, reward, done, _, _ = env.step(action)
            power.append(-reward)
        all_power.append(power)
    return np.mean(all_power, axis=0)


def periodic_sleep_baseline(env, active_duration=5, sleep_duration=15, num_episodes=10):
    """Run a baseline with periodic sleep/wake cycles."""
    all_power = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        power = []
        counter = 0
        state = 0  # 0: active, 1: sleep
        while not done:
            if state == 0 and counter >= active_duration:
                action = 1  # switch to sleep
                state = 1
                counter = 0
            elif state == 1 and counter >= sleep_duration:
                action = 0  # switch to active
                state = 0
                counter = 0
            else:
                action = state  # continue in current state
            obs, reward, done, _, _ = env.step(action)
            power.append(-reward)
            counter += 1
        all_power.append(power)
    return np.mean(all_power, axis=0)


def plot_results(
    dqn_power=None,
    ppo_power=None,
    baseline_power=None,
    random_power=None,
    threshold_power=None,
    periodic_power=None,
    ppo_queue_requests=None,
    dqn_queue_requests=None,
    dqn_transitions=None,
    ppo_transitions=None,
    episode_duration=None,
):
    """Generate result plots."""
    os.makedirs("figs", exist_ok=True)

    # FIGURE 1 - Power Consumption Comparison
    plt.figure(1)

    if dqn_power is not None:
        plt.plot(dqn_power, label="DQN", color="blue")
        rms_dqn = math.sqrt(sum([p**2 for p in dqn_power]) / len(dqn_power))
        plt.axhline(y=rms_dqn, color="blue", linestyle="dotted", linewidth=2)
        plt.text(
            episode_duration - 50,
            rms_dqn - 0.15,
            f"RMS DQN: {rms_dqn:.2f} mW",
            ha="right",
            va="top",
            fontweight="bold",
        )

    if ppo_power is not None:
        plt.plot(ppo_power, label="PPO", color="green")
        rms_ppo = math.sqrt(sum([p**2 for p in ppo_power]) / len(ppo_power))
        plt.axhline(y=rms_ppo, color="green", linestyle="dotted", linewidth=2)
        plt.text(
            episode_duration - 58,
            rms_ppo - 0.15,
            f"RMS PPO: {rms_ppo:.2f} mW",
            ha="right",
            va="top",
            fontweight="bold",
        )

    if baseline_power is not None:
        plt.plot(baseline_power, label="AlwaysActive", color="red", linestyle="-")
        rms_baseline = math.sqrt(
            sum([p**2 for p in baseline_power]) / len(baseline_power)
        )
        plt.axhline(y=rms_baseline, color="red", linestyle="dotted", linewidth=2)
        plt.text(
            episode_duration - 1,
            rms_baseline + 0.35,
            f"RMS AlwaysActive: {rms_baseline:.2f} mW",
            ha="right",
            va="bottom",
            fontweight="bold",
        )
    if random_power is not None:
        plt.plot(
            random_power,
            label="RandAct",
            color="purple",
            linestyle="-",
            linewidth=2,
        )
        rms_random = math.sqrt(sum([p**2 for p in random_power]) / len(random_power))
        plt.axhline(y=rms_random, color="purple", linestyle="dotted", linewidth=2)
        plt.text(
            episode_duration - 1,
            rms_random + 0.35,
            f"RMS RandAct: {rms_random:.2f} mW",
            ha="right",
            va="bottom",
            fontweight="bold",
        )
    if threshold_power is not None:
        plt.plot(
            threshold_power,
            label="Threshold-based",
            color="orange",
            linestyle="-",
        )
        rms_threshold = math.sqrt(
            sum([p**2 for p in threshold_power]) / len(threshold_power)
        )
        plt.axhline(
            y=rms_threshold,
            color="orange",
            linestyle="dotted",
        )
        plt.text(
            episode_duration - 1,
            rms_threshold + 0.1,
            f"RMS Threshold-based: {rms_threshold:.2f} mW",
            ha="right",
            va="bottom",
            fontweight="bold",
        )

    if periodic_power is not None:
        plt.plot(
            periodic_power,
            label="PeriodicSleep",
            color="brown",
            linestyle="-",
        )
        rms_periodic = math.sqrt(
            sum([p**2 for p in periodic_power]) / len(periodic_power)
        )
        plt.axhline(
            y=rms_periodic,
            color="brown",
            linestyle="dotted",
        )
        plt.text(
            episode_duration - 10,
            rms_periodic - 0.15,
            f"RMS PeriodicSleep: {rms_periodic:.2f} mW",
            ha="right",
            va="top",
            fontweight="bold",
        )

    plt.xlim(0, episode_duration)

    plt.ylabel("Power (mW)")
    plt.xlabel("Time Step")
    plt.title("Power Consumption")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3, which="both")
    plt.savefig("figs/power_comparison.png", dpi=300)

    # FIGURE 2 - Requests in Queue Over Time
    if ppo_queue_requests is not None:
        plt.figure(2)
        plt.stem(
            ppo_queue_requests, label="Requests in Queue", linefmt="r-", markerfmt="ko"
        )
        plt.ylabel("Queue length")
        plt.xlabel("Time Step")
        plt.title("PPO: Requests in Queue")
        plt.legend()
        plt.xlim(0, episode_duration)
        plt.grid(alpha=0.3, which="both")
        plt.savefig("figs/ppo_queue_requests.png", dpi=300)

    if dqn_queue_requests is not None:
        plt.figure(3)
        plt.stem(
            dqn_queue_requests, label="Requests in Queue", linefmt="b-", markerfmt="ko"
        )
        plt.ylabel("Queue length")
        plt.xlabel("Time Step")
        plt.title("DQN: Requests in Queue")
        plt.legend()
        plt.xlim(0, episode_duration)
        plt.grid(alpha=0.3, which="both")
        plt.savefig("figs/dqn_queue_requests.png", dpi=300)

    # FIGURE 3 - Interarrivals and Power Mode (DQN)
    if dqn_transitions is not None:
        plt.figure(4)
        go_active = [
            1 if i == "go_active" else 0 for i in dqn_transitions
        ]  # last episode
        plt.plot(go_active, label="Power Mode", drawstyle="steps-pre", color="b")
        plt.plot(env.timeline, label="Interarrivals", color="black", marker="o")
        plt.yticks([0, 1], ("Sleep/0", "Active/1"))
        plt.xlabel("Cycle")
        plt.title("Controller: DQN")
        plt.minorticks_on()
        plt.legend()
        plt.xlim(0, episode_duration)
        plt.grid(alpha=0.3, which="both")
        plt.savefig("figs/dqn_transitions.png", dpi=300)

    # FIGURE 4 - Interarrivals and Power Mode (PPO)
    if ppo_transitions is not None:
        plt.figure(5)
        go_active = [
            1 if i == "go_active" else 0 for i in ppo_transitions
        ]  # last episode
        plt.plot(go_active, label="Power Mode", drawstyle="steps-pre", color="r")
        plt.plot(
            env.timeline,
            label="Interarrivals",
            color="black",
            marker="o",
        )
        plt.yticks([0, 1], ("Sleep/0", "Active/1"))
        plt.xlabel("Cycle")
        plt.title("Controller: PPO")
        plt.legend()
        plt.xlim(0, episode_duration)
        plt.grid(alpha=0.3, which="both")
        plt.savefig("figs/ppo_transitions.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate RL agents.")
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate the saved model(s)."
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["DQN", "PPO", "both"],
        default="both",
        help="Type of RL agent to train or evaluate ('DQN', 'PPO', or 'both').",
    )
    args = parser.parse_args()

    env = SystemEnvironment()
    check_env(env)
    policy_kwargs = {"net_arch": [64, 64]}

    if args.eval_only:
        dqn_power = ppo_power = dqn_transitions = ppo_transitions = None
        # Evaluate DQN agent
        if args.agent in ("DQN", "both") and os.path.exists("logs/dqn_model.zip"):
            dqn_model = DQN.load("logs/dqn_model")
            (
                dqn_transitions,
                dqn_queue_requests,
                dqn_power,
                dqn_rewards,
            ) = evaluate_agent(env, dqn_model, num_episodes=10)
            dqn_power = np.mean(dqn_power, axis=0)
            dqn_transitions = dqn_transitions[-1]  # Take transitions of last episode

        # Evaluate PPO agent
        if args.agent in ("PPO", "both") and os.path.exists("logs/ppo_model.zip"):
            ppo_model = PPO.load("logs/ppo_model")
            (
                ppo_transitions,
                ppo_queue_requests,
                ppo_power,
                ppo_rewards,
            ) = evaluate_agent(env, ppo_model, num_episodes=10)
            ppo_power = np.mean(ppo_power, axis=0)
            ppo_transitions = ppo_transitions[-1]  # Take transitions of last episode

        # Check if at least one model was evaluated
        if dqn_power is None and ppo_power is None:
            raise FileNotFoundError(
                "No saved models found for evaluation. Please train a model first."
            )

        # Generate baselines
        baseline_power = always_active_baseline(env, num_episodes=10)
        random_power = random_action_baseline(env, num_episodes=10)
        threshold_power = threshold_based_sleep_baseline(env, num_episodes=10)
        periodic_power = periodic_sleep_baseline(env, num_episodes=10)

        # Plot results
        plot_results(
            dqn_power=dqn_power,
            ppo_power=ppo_power,
            baseline_power=baseline_power,
            random_power=random_power,
            threshold_power=threshold_power,
            periodic_power=periodic_power,
            ppo_queue_requests=ppo_queue_requests[-1],
            dqn_queue_requests=dqn_queue_requests[-1],
            dqn_transitions=dqn_transitions,
            ppo_transitions=ppo_transitions,
            episode_duration=cfg.num_steps,
        )

    else:
        # Train the agent
        if args.agent in ("DQN", "both"):
            dqn_model = train_agent(
                env, cfg.total_timesteps, policy_kwargs, agent_type="DQN"
            )
            dqn_model.save("logs/dqn_model")

        if args.agent in ("PPO", "both"):
            ppo_model = train_agent(
                env, cfg.total_timesteps, policy_kwargs, agent_type="PPO"
            )
            ppo_model.save("logs/ppo_model")
