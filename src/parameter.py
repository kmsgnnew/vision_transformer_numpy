import copy

import numpy as np
from optimizer import Optimizer


class Parameter:
    """Parameter wrapper to handle cls."""

    def __init__(self, val) -> None:
        """Initialize."""
        self.val = val
        self.optimizer = None

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.
        """
        self.cache = dict(grad=np.sum(grad, axis=0)[None, :])

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        cloned_optimizer = copy.deepcopy(optimizer)
        self.optimizer = cloned_optimizer

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.val = self.optimizer.update(self.cache["grad"], self.val)
