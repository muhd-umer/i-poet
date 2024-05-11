"""
Gymnasium (gym) environment for the system
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..config import get_default_cfg
from ..structs import IoTQueue
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
        state_size = (
            len(self.controller.states) + len(self.node.states) + len(self.queue.states)
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_size, 2), dtype=np.float32
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
        return np.array(
            [
                self.controller.current_state,
                self.node.current_state,
                self.queue.current_state,
            ]
        )

    def _get_info(self):
        """
        Get the information
        """
        return {
            "controller": self.controller.current_state,
            "node": self.node.current_state,
            "queue": self.queue.current_state,
        }

    def cost_function(self, virtual_state=False, delta=0.0):
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
        if virtual_state:
            cnt_state = self.controller.current_state
            queue_state = self.queue.current_state

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

                    return (power_a2s * time_a2s + power_s2a * time_s2a) / 2.0

        else:
            for st in cnt_sm:
                if st["state"]["power_mode"] == cnt_state:
                    power_cost = st["state"]["power"]

            performance_penalty = queue_state

            return power_cost + delta * performance_penalty

    def step(self, action):
        """
        Take a step in the environment
        """
        if self.time in self.inter_arrivals:
            reqs = self.node.generate_requests(np.random.randint(1, 8))
            self.queue.allocate_space(reqs)
            self.requests_arrival.append(reqs)
        else:
            self.requests_arrival.append(0)

        if self.queue.is_deque_ready and self.controller.current_state == "active":
            self.queue.dequeue_request()
        else:
            self.queue.enqueue_request()

        self.node.determine_state()

    def reset(self):
        """
        Reset the environment
        """
        self._init_system()
        self.requests_arrival = []

        return self._get_observation(), self._get_info()
