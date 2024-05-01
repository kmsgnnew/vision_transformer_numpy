import sys

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

sys.path.append(".")
import unittest

from src.layer_normalization import LayerNormalization
from utils import access_grad


class TestModel(nn.Module):
    """Sample model under test."""

    def __init__(self):
        """Initialize."""
        super(TestModel, self).__init__()
        self.mlp_1 = nn.Linear(12, 8)
        self.ln = nn.LayerNorm(8)
        self.mlp_3 = nn.Linear(8, 10)

    def ln_grad_out_hook(self, grad):
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_out = grad

    def ln_grad_in_hook(self, grad):
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_in = grad

    def forward(self, images):
        """Forward propagation.

        Args:
            images: images.

        Returns:
            output of forward propagation.
        """
        out1 = self.mlp_1(images)
        out1.register_hook(self.ln_grad_out_hook)
        out2 = self.ln(out1)
        out2.register_hook(self.ln_grad_in_hook)
        out3 = self.mlp_3(out2)
        return out1, out2, out3


class TestLayerNormalization(unittest.TestCase):
    def test_linear(self):
        """Test functioning of Layer Normalization."""
        model = TestModel()
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()
        # create a random gt
        x = torch.tensor(np.random.rand(2, 50, 12), dtype=torch.float, requires_grad=True)
        y_ = np.zeros((2, 10))
        y_[0][5] = 1
        y_[1][2] = 1
        y = torch.tensor(y_, dtype=torch.long)
        # infer model
        out1, out2, y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        mapped_grad, _ = access_grad(model)

        # assign
        input = out1.detach().numpy()
        output = out2.detach().numpy()
        grad_in = model.grad_in.detach().numpy()
        grad_out = model.grad_out.detach().numpy()
        # call our layer normalization
        grad_gamma = mapped_grad["ln.weight"].detach().numpy()
        grad_beta = mapped_grad["ln.bias"].detach().numpy()
        custom_ln = LayerNormalization(8)
        custom_out = custom_ln.forward(input)
        decimal_place = 3
        message = "NumPy and reference implementation not almost equal."
        np.testing.assert_array_almost_equal(custom_out, output, decimal_place, message)
        custom_grad_out = custom_ln.backward(grad_in)
        np.testing.assert_array_almost_equal(custom_grad_out, grad_out, decimal_place, message)
        custom_grad_gamma, custom_grad_beta = custom_ln.get_grads()
        np.testing.assert_array_almost_equal(custom_grad_gamma, grad_gamma, decimal_place, message)
        np.testing.assert_array_almost_equal(custom_grad_beta, grad_beta, decimal_place, message)


if __name__ == "__main__":
    unittest.main()
