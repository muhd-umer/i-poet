"""
Main script to train a RL agents on the system environment and evaluate performance;
the evaluation results are plotted to visualize the power consumption, queue requests,
and power mode transitions over time.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from config import get_default_cfg
from system import SystemEnvironment

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
    )
    model.learn(total_timesteps=total_timesteps)

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
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        transitions = []
        queue_requests = []
        power = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            transitions.append(env.controller.current_action)
            queue_requests.append(env.queue.current_state)
            power.append(-reward)  # Reward is negative cost
        all_transitions.append(transitions)
        all_queue_requests.append(queue_requests)
        all_power.append(power)
    return all_transitions, all_queue_requests, all_power


def plot_results(transitions, queue_requests, power, inter_arrivals, num_steps):
    """Generate and display plots based on the evaluation results.

    Args:
        transitions (list): List of lists containing the controller's actions for
            each episode.
        queue_requests (list): List of lists containing the queue size for each
            timestep of each episode.
        power (list): List of lists containing the power consumption for each
            timestep of each episode.
        inter_arrivals (list): List of inter-arrival times.
        num_steps (int): Number of timesteps in each episode.
    """
    plt.figure(1)
    plt.plot(inter_arrivals, label="Interarrivals", color="black", marker="o")
    # Average power mode over all episodes
    go_active = np.mean(
        [[1 if i == "go_active" else 0 for i in ep] for ep in transitions], axis=0
    )
    plt.plot(go_active, label="Power Mode")
    plt.yticks([0, 1], ("Sleep", "Active"))
    plt.ylabel("Power Mode / Interarrivals")
    plt.xlabel("Cycle")
    plt.title("Service Provider - Power Mode")
    plt.minorticks_on()
    plt.legend()

    plt.figure(2)
    # RMS queue requests over all episodes
    rms_queue_requests = [math.sqrt(np.mean(np.square(q))) for q in queue_requests]
    plt.plot(
        rms_queue_requests,
        label="RMS Requests In Queue",
        color="green",
    )
    plt.ylabel("Number Of Requests")
    plt.xlabel("Episode")
    plt.title("Service Queue - RMS Requests")
    plt.minorticks_on()
    plt.xlabel("Episode")
    plt.legend()

    plt.figure(3)
    # RMS power consumption over all episodes
    rms_power = [math.sqrt(np.mean(np.square(p))) for p in power]
    plt.plot(
        rms_power,
        label="RMS Power - (DPM)",
    )
    plt.ylabel("(mW)")
    plt.xlabel("Episode")
    plt.title("Service Provider - RMS Power Cost ")
    plt.minorticks_on()
    plt.legend()

    plt.show()


def main():
    """Main function."""
    env = SystemEnvironment()
    check_env(env)

    policy_kwargs = {"net_arch": [64, 64]}

    # Train the agent
    trained_model = train_dqn(env, cfg.total_timesteps, policy_kwargs)
    trained_model.save("logs/dqn_model")

    # Evaluate the trained agent
    transitions, queue_requests, power = evaluate_agent(
        env, trained_model, num_episodes=10
    )

    # Plot the results
    plot_results(transitions, queue_requests, power, cfg.inter_arrivals, cfg.num_steps)


if __name__ == "__main__":
    main()
