"""
Controller module for power optimization system for IoT devices

This module contains the Controller class that manages the power mode of a
device; it is used to set the power mode based on the given command.

The Controller class has the following methods:
    __init__: Initialize the Controller with the given state machine and transfer rate.
    init_states: Initialize the states from the state machine.
    observe_next: Return the next state based on the given command.
    set_power_mode: Set the power mode based on the given command.
"""


class Controller:
    """
    Represents a Controller that manages the power mode of a device.
    """

    def __init__(self, state_machine, transfer_rate):
        """Initialize the Controller with the given state machine and transfer rate.

        Args:
            state_machine (list): State machine of the device.
            transfer_rate (float): Transfer rate of the communication module.
        """
        self.state_machine = state_machine
        self.transfer_rate = transfer_rate
        self.current_state = self.previous_state = self.next_state = (
            self.current_action
        ) = self.previous_action = None
        self.init_states()

    def init_states(self):
        """Initialize the states from the state machine."""
        for state in self.state_machine:
            if state["state"]["init"]:
                self.current_state = state["state"]["power_mode"]
                self.current_action = state["state"]["command"]
        self.previous_state = self.next_state = self.current_state
        self.previous_action = self.current_action

    def observe_next(self, command):
        """Return the next state based on the given command.

        Args:
            command (str): The command to execute.

        Returns:
            str: The next state.
        """
        for state in self.state_machine:
            sp_commands = state["state"]["command"]
            sp_state = state["state"]["power_mode"]

            if command == sp_commands:
                return "sleep" if sp_state == "active" else "active"

    def set_power_mode(self, command):
        """Set the power mode based on the given command.

        Args:
            command (str): The command to execute.
        """
        if self.current_state == "transient":
            self.previous_state = self.current_state
            self.current_state = self.next_state
            return

        self.previous_action = self.current_action
        self.current_action = command

        state_transitions = {
            "active": {"go_sleep": "sleep", "go_active": "active"},
            "sleep": {"go_active": "active", "go_sleep": "sleep"},
        }

        if command in state_transitions[self.current_state]:
            self.next_state = state_transitions[self.current_state][command]
            self.current_state = (
                "transient"
                if self.next_state != self.current_state
                else self.next_state
            )
            self.previous_state = self.current_state
