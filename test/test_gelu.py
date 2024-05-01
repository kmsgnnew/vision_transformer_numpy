import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

sys.path.append(".")
from typing import Tuple

from src.gelu import GELU


class TestModel(nn.Module):
    """Sample model under test"""

    def __init__(self) -> None:
        """Initialize."""
        super(TestModel, self).__init__()
        self.mlp_1 = nn.Linear(12, 8)
        self.gelu = nn.GELU()
        self.mlp_3 = nn.Linear(8, 10)

    def gelu_layer_grad_out_hook(self, grad: np.ndarray) -> None:
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_out = grad

    def gelu_layer_grad_in_hook(self, grad: np.ndarray) -> None:
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_in = grad

    def forward(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation.

        Args:
            images: images.

        Returns:
            output of forward propagation.
        """
        out_1 = self.mlp_1(images)
        out_1.register_hook(self.gelu_layer_grad_out_hook)
        out_2 = self.gelu(out_1)
        out_2.register_hook(self.gelu_layer_grad_in_hook)
        out_3 = self.mlp_3(out_2)
        return out_1, out_2, out_3


class TestGelu(unittest.TestCase):
    def test_gelu(self) -> None:
        """Test functioning of gelu."""
        model = TestModel()
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # create gt
        x = torch.tensor(np.random.rand(2, 50, 12), dtype=torch.float, requires_grad=True)
        y_ = np.zeros((2, 10))
        y_[0][5] = 1
        y_[1][2] = 1
        y = torch.tensor(y_, dtype=torch.long)

        # infer result and calculate grad in pytorch
        out_1, out_2, y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()

        ## assign
        input = out_1.detach().numpy()
        output = out_2.detach().numpy()
        grad_in = model.grad_in.detach().numpy()
        grad_out = model.grad_out.detach().numpy()

        # call our implementation
        custom_gelu = GELU()
        custom_out = custom_gelu.forward(input)
        custom_grad_out = custom_gelu.backward(grad_in)

        decimal_place = 3
        message = "NumPy and reference implementation not almost equal."
        np.testing.assert_array_almost_equal(custom_out, output, decimal_place, message)
        np.testing.assert_array_almost_equal(custom_grad_out, grad_out, decimal_place, message)


if __name__ == "__main__":
    unittest.main()
