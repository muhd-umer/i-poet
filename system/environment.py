"""
Gymnasium (gym) environment for the system
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import get_default_cfg
from structs import IoTQueue

from .controller import Controller
from .node import IoTNode

cfg = get_default_cfg()
gym.logger.set_level(40)


class SystemEnvironment(gym.Env):
    """
    A gym environment for the system
    """

    ACTION_MAP = {0: "go_active", 1: "go_sleep"}

    def __init__(self, cfg=cfg):
        super().__init__()

        self.state_machine = cfg.cnt_sm
        self.transfer_rate = cfg.transfer_rate
        self.queue_size = cfg.queue_size
        self.valid_reqs = cfg.valid_reqs
        self.delta = cfg.delta

        self._init_system()
        self.inter_arrivals = cfg.inter_arrivals
        self.timeline = [
            1 if i in self.inter_arrivals else 0 for i in range(0, cfg.num_steps)
        ]
        self.requests_arrival = []
        self.time = 0

        # action space
        self.action_space = spaces.Discrete(2)

        # observation space
        by2bi = 8  # 1 byte = 8 bits
        self.observation_space = spaces.Dict(
            {
                "controller_state": spaces.Discrete(len(self.controller.states)),
                "node_state": spaces.Discrete(len(self.node.states)),
                "queue_state": spaces.Discrete(self.queue_size + 1),
                "requests": spaces.Discrete(int(by2bi * 8 / self.transfer_rate)),
            }
        )

    def _init_system(self):
        """
        Initialize the system
        """
        self.controller = Controller(state_machine=self.state_machine)
        self.node = IoTNode(states=self.valid_reqs, transfer_rate=self.transfer_rate)
        self.queue = IoTQueue(size=self.queue_size)

    def _get_observation(self):
        """
        Get the observation
        """

        obs = {
            "controller_state": self.controller.states.index(
                self.controller.current_state
            ),
            "node_state": self.node.states.index(self.node.current_state),
            "queue_state": self.queue.current_state,
            "requests": self.node.requests,
        }

        return obs

    def _get_info(self):
        """
        Get the information
        """
        return {
            "controller": self.controller.current_state,
            "node": self.node.current_state,
            "queue": self.queue.current_state,
        }

    def cost_function(self, delta=1.0):
        """
        Cost function for the environment

        Args:
            virtual_state (bool): Flag to use the virtual state.
            delta (float): The performance penalty.

        Returns:
            float: The cost of the environment.
        """

        cnt_sm = self.controller.state_machine
        cnt_state = self.controller.previous_state
        queue_state = self.queue.previous_state

        if cnt_state == "transient":
            for st in cnt_sm:
                if st["state"]["power_mode"] == "transient":
                    power_a2s = 0
                    time_a2s = st["state"]["transient_timing"]["s2a"]
                    power_s2a = 0
                    time_s2a = st["state"]["transient_timing"]["a2s"]

                    if st["state"]["power_mode"] == "active":
                        power_s2a = st["state"]["power"]

                    elif st["state"]["power_mode"] == "sleep":
                        power_a2s = st["state"]["power"]

                    transient_power = (
                        power_a2s * time_a2s + power_s2a * time_s2a
                    ) / 2.0
                    transition_penalty = 0.1 * (
                        time_a2s + time_s2a
                    )  # Transition penalty

                    return transient_power + transition_penalty

        else:
            for st in cnt_sm:
                if st["state"]["power_mode"] == cnt_state:
                    power_cost = st["state"]["power"]

            performance_penalty = queue_state

            return power_cost + delta * performance_penalty

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action (int): The action to take.

        Returns:
            tuple: The observation, reward, done flag, truncated flag, and info.
        """
        action = self.ACTION_MAP[action]
        self.controller.set_power_mode(action)
        self.time += 1

        if self.time in self.inter_arrivals:
            reqs = self.node.generate_requests(self.np_random.integers(1, 8, 1)[0])
            self.node.requests = reqs
            self.queue.allocate_space(reqs)
            self.requests_arrival.append(reqs)
        else:
            self.requests_arrival.append(0)

        if self.queue.is_deque_ready and self.controller.current_state == "active":
            self.queue.dequeue_request()
        else:
            self.queue.enqueue_request()

        self.node.determine_state()

        done = True if self.time >= cfg.num_steps else False

        reward = -1 * self.cost_function(delta=self.delta)  # minimize the cost

        return self._get_observation(), reward, done, False, self._get_info()

    def reset(self, seed=None, options=None):
        """
        Reset the environment

        Args:
            seed (int, optional): The seed for the random number generator. Defaults
                to None.
            options (dict, optional): Additional options for resetting the
                environment. Defaults to None.

        Returns:
            tuple: The observation and info.
        """
        super().reset(seed=seed)

        self._init_system()
        self.requests_arrival = []
        self.time = 0

        return self._get_observation(), self._get_info()
