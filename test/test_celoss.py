import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss as CrossEntropyLossRef
from torch.optim import Adam

sys.path.append(".")

from src.cross_entropy_loss import CrossEntropyLoss


class TestModel(nn.Module):
    """Sample model under test"""

    def __init__(self) -> None:
        """Initialize."""
        super(TestModel, self).__init__()
        self.mlp_1 = nn.Linear(12, 8)
        self.mlp_2 = nn.Linear(8, 10)

    def ce_grad_out_hook(self, grad: np.ndarray) -> None:
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_out = grad

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            images: images.

        Returns:
            output of forward propagation.
        """
        out_1 = self.mlp_1(images)
        out_2 = self.mlp_2(out_1)
        out_2.register_hook(self.ce_grad_out_hook)
        return out_2


class TestCrossEntropy(unittest.TestCase):
    def test_ce(self) -> None:
        """Test functioning of cross entropy loss."""
        model = TestModel()
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLossRef()

        # create a gt
        x = torch.tensor(np.random.rand(3, 12), dtype=torch.float, requires_grad=True)
        y_ = np.zeros((3))
        y_[0] = 1
        y_[1] = 3
        y_[2] = 5
        y = torch.tensor(y_, dtype=torch.long)

        # infer reference implementation
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()

        # access outputs
        input = y_hat.detach().numpy()
        output = loss.detach().numpy()
        grad_out = model.grad_out.detach().numpy()

        # call our cross entropy
        custom_ce = CrossEntropyLoss()
        decimal_place = 3
        message = "NumPy and reference implementation not almost equal."
        custom_out = custom_ce.forward(input, np.array([1, 3, 5]))

        # validate
        np.testing.assert_array_almost_equal(custom_out, output, decimal_place, message)
        custom_grad_out = custom_ce.backward()
        np.testing.assert_array_almost_equal(custom_grad_out, grad_out, decimal_place, message)


if __name__ == "__main__":
    unittest.main()
