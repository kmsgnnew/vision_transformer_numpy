import numpy as np


class GELU:
    """Computes GELU activation function."""

    def __init__(self) -> None:
        """Initialize."""
        self.cache = dict(input=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed GELU output.
        """
        # this is an approximation. tanh based implementation is more accurate
        # z = x * self.sigmoid(1.702 * x)
        self.cache = dict(input = x)
        z = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))
        return z

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation.

        Args:
            x: input array.

        Returns:
            computed sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        x = self.cache["input"]
        # ref - https://github.com/AkiRusProd/numpy-transformer/blob/master/transformer/activations.py
        sech = lambda z: 2 / (np.exp(z) + np.exp(-z))
        return grad * (
            0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + (0.0535161 * np.power(x, 3) + 0.398942 * x) * np.power(sech(0.0356774 * np.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        ).astype(x.dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            x: input array.

        Returns:
            computed GELU output.
        """
        return self.forward(x)
