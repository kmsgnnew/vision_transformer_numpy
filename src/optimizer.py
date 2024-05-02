from abc import abstractmethod

import numpy as np


class Optimizer:
    """Optimizer."""

    @abstractmethod
    def update(self, grad: np.ndarray, w: np.ndarray) -> None:
        """Update weights based on gradient.

        Args:
            grad: gradient.
            w: weights to be updated.
        """
        pass

class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, learning_rate: float = 0.001) -> None:
        """Initialize.

        Args:
            learning_rate: learning rate. Defaults to 0.001.
        """
        self.learning_rate = learning_rate

    def update(self, grad: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Update weights based on gradient.

        Args:
            grad: gradient.
            w: weights to be updated.

        Returns:
            updated weights.
        """
        w -= grad * self.learning_rate
        return w


class Adam(Optimizer):
    """Implements Adam optimizer."""

    def __init__(
        self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8
    ) -> None:
        """Initialize

        Args:
            learning_rate: learning rate. Defaults to 0.01.
            beta1: beta 1. Defaults to 0.9.
            beta2: beta 2. Defaults to 0.999.
            epsilon: epsilon. Defaults to 1e-8.
        """
        self.m_dw, self.rms_dw = 0, 0
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1

    def update(self, grad: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Update weights based on gradient.

        Args:
            grad: gradient.
            w: weights to be updated.

        Returns:
            updated weights.
        """
        # momentum calc with beta 1
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * grad

        # rms calculation with beta 2
        self.rms_dw = self.beta2 * self.rms_dw + (1 - self.beta2) * (grad**2)

        # bias correction
        m_dw_corr = self.m_dw / (1 - (self.beta1**self.t))
        rms_dw_corr = self.rms_dw / (1 - (self.beta2**self.t))

        # update weights
        w = w - self.learning_rate * (m_dw_corr / (np.sqrt(rms_dw_corr) + self.epsilon))
        return w
