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

        # action space
        self.action_space = spaces.Discrete(2)

        # observation space
        # state_size =

    def _init_system(self):
        """
        Initialize the system
        """
        self.controller = Controller(state_machine=self.state_machine)
        self.node = IoTNode(states=self.valid_reqs, transfer_rate=self.transfer_rate)
        self.queue = IoTQueue(size=self.queue_size)
