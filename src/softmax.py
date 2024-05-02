import numpy as np


class Softmax:
    """Computes softmax."""

    def __init__(self) -> None:
        """Initialize."""
        self.cache = dict(output=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed softmax output.
        """
        max_val = np.max(x, axis=-1)[:, None]  # for numerical stability
        y = np.exp(x - max_val) / np.sum(np.exp(x - max_val), axis=-1, keepdims=True)
        self.cache = dict(output=y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        softmax = self.cache["output"]
        # ref - https://github.com/tensorflow/tensorflow/blob/0.5.0/tensorflow/python/ops/nn_grad.py
        # fails
        # return softmax * (grad -(grad * softmax).sum(axis=1)[:,None])
        # ref - https://github.com/AkiRusProd/numpy-transformer/blob/master/transformer/activations.py
        J = softmax[..., np.newaxis] * np.tile(
            np.identity(softmax.shape[-1]), (softmax.shape[0], *tuple(np.ones(softmax.ndim, dtype=np.int8).tolist()))
        ) - (
            softmax[..., np.newaxis, :].transpose(
                *tuple(np.arange(0, softmax.ndim - 1, 1, dtype=np.int8).tolist()), -1, -2
            )
            @ softmax[..., np.newaxis, :]
        )
        input_grad = grad[..., np.newaxis, :] @ J
        return input_grad.reshape(grad.shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            x: input array.

        Returns:
            computed softmax output.
        """
        return self.forward(x)
