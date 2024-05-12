"""
Main script to train RL agents on the multi-node system environment
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from config import get_extended_cfg
from system import MultiNodeEnvironment

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

# increase font size
plt.rcParams.update({"font.size": 11})

cfg = get_extended_cfg()


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
            tensorboard_log="./extended_logs/",
            policy_kwargs=policy_kwargs,
            learning_rate=2.3e-4,
            batch_size=256,
            buffer_size=10000,
            train_freq=4,
            gamma=0.9,
            seed=1703,
        )
    elif agent_type == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./extended_logs/",
            policy_kwargs=policy_kwargs,
            learning_rate=7.5e-4,
            n_steps=200,
            batch_size=200,
            normalize_advantage=False,
            ent_coef=1e-6,
            clip_range=0.2,
            gae_lambda=0.95,
            n_epochs=12,
            gamma=0.9,
            seed=1703,
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


def plot_results(
    dqn_power=None,
    ppo_power=None,
    baseline_power=None,
    ppo_queue_requests=None,
    dqn_queue_requests=None,
    dqn_transitions=None,
    ppo_transitions=None,
    episode_duration=None,
):
    """Generate result plots."""
    os.makedirs("extended_figs", exist_ok=True)

    # FIGURE 1 - Power Consumption Comparison
    plt.figure(1)

    if dqn_power is not None:
        plt.plot(dqn_power, label="DQN", color="blue")
        rms_dqn = math.sqrt(sum([p**2 for p in dqn_power]) / len(dqn_power))
        plt.axhline(y=rms_dqn, color="blue", linestyle="dotted", label="RMS DQN")
        plt.text(
            episode_duration - 1,
            rms_dqn + 0.1,
            f"RMS DQN: {rms_dqn:.2f} mW",
            ha="right",
            va="bottom",
        )

    if ppo_power is not None:
        plt.plot(ppo_power, label="PPO", color="green")
        rms_ppo = math.sqrt(sum([p**2 for p in ppo_power]) / len(ppo_power))
        plt.axhline(y=rms_ppo, color="green", linestyle="dotted", label="RMS PPO")
        plt.text(
            episode_duration - 1,
            rms_ppo - 0.15,
            f"RMS PPO: {rms_ppo:.2f} mW",
            ha="right",
            va="top",
        )

    if baseline_power is not None:
        plt.plot(baseline_power, label="Always Active", color="red", linestyle="dashed")
        rms_baseline = math.sqrt(
            sum([p**2 for p in baseline_power]) / len(baseline_power)
        )
        plt.axhline(
            y=rms_baseline, color="red", linestyle="dotted", label="RMS Always Active"
        )
        plt.text(
            episode_duration - 1,
            rms_baseline + 0.1,
            f"RMS Always Active: {rms_baseline:.2f} mW",
            ha="right",
            va="bottom",
        )

    plt.xlim(0, episode_duration)

    plt.ylabel("Power (mW)")
    plt.xlabel("Time Step")
    plt.title("Power Consumption")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("extended_figs/power_comparison.png", dpi=300)

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
        plt.grid(alpha=0.3)
        plt.savefig("extended_figs/ppo_queue_requests.png", dpi=300)

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
        plt.grid(alpha=0.3)
        plt.savefig("extended_figs/dqn_queue_requests.png", dpi=300)

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
        plt.grid(alpha=0.3)
        plt.savefig("extended_figs/dqn_transitions.png", dpi=300)

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
        plt.grid(alpha=0.3)
        plt.savefig("extended_figs/ppo_transitions.png", dpi=300)

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

    env = MultiNodeEnvironment()
    check_env(env)
    policy_kwargs = {"net_arch": [64, 64]}

    if args.eval_only:
        dqn_power = ppo_power = dqn_transitions = ppo_transitions = None
        # Evaluate DQN agent
        if args.agent in ("DQN", "both") and os.path.exists(
            "extended_logs/dqn_model.zip"
        ):
            dqn_model = DQN.load("extended_logs/dqn_model")
            (
                dqn_transitions,
                dqn_queue_requests,
                dqn_power,
                dqn_rewards,
            ) = evaluate_agent(env, dqn_model, num_episodes=10)
            dqn_power = np.mean(dqn_power, axis=0)
            dqn_transitions = dqn_transitions[-1]  # Take transitions of last episode

        # Evaluate PPO agent
        if args.agent in ("PPO", "both") and os.path.exists(
            "extended_logs/ppo_model.zip"
        ):
            ppo_model = PPO.load("extended_logs/ppo_model")
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

        # Generate baseline
        baseline_power = always_active_baseline(env, num_episodes=10)

        # Plot results
        plot_results(
            dqn_power=dqn_power,
            ppo_power=ppo_power,
            baseline_power=baseline_power,
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
            dqn_model.save("extended_logs/dqn_model")

        if args.agent in ("PPO", "both"):
            ppo_model = train_agent(
                env, cfg.total_timesteps, policy_kwargs, agent_type="PPO"
            )
            ppo_model.save("extended_logs/ppo_model")
