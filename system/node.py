"""Interface for Nodes (requestors); used to request services of the IoT devices

This module contains the IoTNode class that represents a node in the IoT system.

The IoTNode class has the following methods:
    __init__: Initialize the IoTNode with the given states and alpha value.
    _get_ema_bounds: Calculate the EMA bounds for a given state index.
    determine_state: Determine the current state based on the EMA of the
        requests.
"""


class IoTNode:
    """IoTNode class for representing a node in the IoT system.

    Determines the state of the node based on the number of requests and EMA.
    """

    def __init__(self, states: list, alpha: float = 1.0):
        """Initializes IoTNode with the given states and alpha value.

        Args:
            states (list): List of states.
            alpha (float): Alpha value for EMA calculation.
        """
        self.states = states
        self.alpha = alpha
        self.ema = 0
        self.requests = 0
        self.current_state = states[0]
        self.previous_state = self.current_state

    def _get_ema_bounds(self, state_index):
        """Calculates the EMA bounds for a given state index.

        Args:
            state_index (int): Index of the state.

        Returns:
            tuple: Lower and upper EMA bounds.
        """
        N = len(self.states)
        x = ((-1 * N) / 2) + state_index  # exponent
        return pow(2, x) * self.ema, pow(2, (x + 1)) * self.ema

    def determine_state(self):
        """Determines the current state based on the EMA of the requests."""
        self.ema = self.alpha * self.requests + self.ema * (1 - self.alpha)
        self.previous_state = self.current_state

        lowest_lower_bound = 0
        for _state in self.states:
            i = self.states.index(_state)
            lower_bound, upper_bound = self._get_ema_bounds(state_index=i)

            if i == 0:
                lowest_lower_bound = lower_bound

            if lower_bound < self.requests < upper_bound:
                self.current_state = _state
            elif i == (len(self.states) - 1) and self.requests > upper_bound:
                self.current_state = _state
            elif i == (len(self.states) - 1) and self.requests <= lowest_lower_bound:
                self.current_state = self.states[0]
