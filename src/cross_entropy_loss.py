import sys

import numpy as np

sys.path.append(".")
from src.softmax import Softmax


class CrossEntropyLoss:
    """Computes the cross entropy loss between input logits and target"""

    def __init__(self) -> None:
        """Initialize."""
        self.softmax = Softmax()
        self.cache = dict(input=None)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            y_pred: input prediction.
            y_true: input target.

        Returns:
            computed cross entropy loss.
        """
        self.cache = dict(input=y_pred)
        one_hot = np.zeros(shape=(len(y_pred), len(y_pred[0])))
        one_hot[np.arange(len(y_pred)), y_true] = 1
        y_pred = np.log(self.softmax(y_pred))
        loss = -y_pred
        self.cache["one_hot"] = one_hot
        out = np.mean(np.sum(loss * one_hot, axis=1))
        return out

    def backward(self) -> np.ndarray:
        """Backward propagation.

        Returns:
            the gradients w.r.t. the input.
        """
        # ref - tensorflow gen_nn_ops.py and nn_grad.py
        input = self.cache["input"]
        one_hot = self.cache["one_hot"]
        return (np.exp(input) / np.sum(np.exp(input), axis=-1, keepdims=True) - one_hot) * (1 / input.shape[-2])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Defining __call__ method to enable function like call.

        Args:
            y_pred: input prediction.
            y_true: input target.

        Returns:
            computed cross entropy output.
        """
        return self.forward(y_pred, y_true)
