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
            obs, reward, done, _, info = env.step(0)
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
        transitions, queue_requests, power, rewards = evaluate_agent(
            env, model, num_episodes=10
        )

        # Generate baseline
        baseline_power = always_active_baseline(env, num_episodes=10)

    else:
        # Train the agent
        trained_model = train_dqn(env, cfg.total_timesteps, policy_kwargs)
        trained_model.save(args.model_path)
