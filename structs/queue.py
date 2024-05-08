"""
Service queue data structure

This module contains the IoTQueue class, which represents a queue data structure
used in the IoT system.

The IoTQueue class has the following methods:
    __init__: Initialize the IoTQueue with the given maximum size.
    current_state: Return the current state of the queue.
    space_available: Return the available space in the queue.
    state_prime: Return the prime state of the queue.
    allocate_space: Allocate space for the requests in the queue.
    enqueue_request: Enqueue a request in the queue.
    dequeue_request: Dequeue a request from the queue.
"""

from collections import deque


class IoTQueue:
    """
    A class used to represent an IoT Queue.

    Attributes:
        queue (deque): a deque object with a maximum length.
        requests (int): the number of requests in the queue.
        previous_state (int): the previous state of the queue.
        is_deque_ready (bool): a flag indicating if the deque is ready.
    """

    def __init__(self, max_size):
        """Constructs all the necessary attributes for the IoTQueue object.

        Args:
            max_size (int): The maximum size of the queue.
        """

        self.queue = deque(maxlen=max_size)
        self.requests = 0
        self.previous_state = self.current_state
        self.is_deque_ready = False

    @property
    def current_state(self):
        """Returns the current state of the queue."""
        return len(self.queue)

    @property
    def space_available(self):
        """Returns the available space in the queue."""
        return self.queue.maxlen - len(self.queue)

    def state_prime(self):
        """Returns the prime state of the queue."""
        if self.is_deque_ready and len(self.queue) > 0:
            return len(self.queue) - 1
        elif not self.is_deque_ready and self.space_available > 0:
            return len(self.queue) + 1
        else:
            return 0

    def allocate_space(self, requests):
        """Allocates space for the requests in the queue.

        Args:
            requests (int): The number of requests to be allocated.

        Returns:
            bool: True if space was allocated, False otherwise.
        """

        if requests <= self.space_available:
            self.requests += requests
            return True
        else:
            return False

    def enqueue_request(self):
        """Enqueues a request in the queue.

        Returns:
            bool: True if the deque is ready, False otherwise.
        """

        self.previous_state = len(self.queue)
        self.is_deque_ready = False
        if self.requests > 0 and self.space_available > 0:
            self.queue.append(1)
            self.requests -= 1
        if self.requests == 0:
            self.is_deque_ready = True
        return self.is_deque_ready

    def dequeue_request(self):
        """Dequeues a request from the queue.

        Returns:
            bool: True if the deque is ready, False otherwise.
        """

        self.previous_state = len(self.queue)
        self.is_deque_ready = True
        if len(self.queue) > 0:
            self.queue.popleft()
            if len(self.queue) == 0:
                self.is_deque_ready = False
        return self.is_deque_ready
