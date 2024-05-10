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

    def __init__(self, states, alpha=1.0, transfer_rate=2.0):
        """Initializes IoTNode with the given states and alpha value.

        Args:
            states (list): List of states.
            alpha (float): Alpha value for EMA calculation.
            transfer_rate (float): Transfer rate of the node.
        """
        self.states = states
        self.alpha = alpha
        self.ema = 0
        self.transfer_rate = transfer_rate
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

    def generate_requests(self, req_size):
        """Generates requests for the node based on the transfer rate.

        Args:
            req_size (int): Size of the request in MB.
        """

        by2bi = 8  # 1 byte = 8 bits
        return req_size * by2bi / self.transfer_rate

    def determine_state(self):
        """Determines the current state based on the EMA of the requests."""
        self.ema = self.alpha * self.requests + self.ema * (1 - self.alpha)
        self.previous_state = self.current_state

        lowest_lower_bound = 0
        for state in self.states:
            i = self.states.index(state)
            lower_bound, upper_bound = self._get_ema_bounds(i)

            if i == 0:
                lowest_lower_bound = lower_bound

            if lower_bound < self.requests < upper_bound:
                self.current_state = state
            elif i == (len(self.states) - 1) and self.requests > upper_bound:
                self.current_state = state
            elif i == (len(self.states) - 1) and self.requests <= lowest_lower_bound:
                self.current_state = self.states[0]
