import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

sys.path.append(".")
from src.linear import Linear
from utils import access_grad


class TestModel(nn.Module):
    """Sample model under test."""

    def __init__(self):
        """Initialize."""
        super(TestModel, self).__init__()
        self.mlp_1 = nn.Linear(12, 8)
        self.linear_layer = nn.Linear(8, 9)
        self.mlp_3 = nn.Linear(9, 10)

    def linear_layer_grad_out_hook(self, grad) -> None:
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_out = grad

    def linear_layer_grad_in_hook(self, grad) -> None:
        """Hook to tap grad in

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
        out_1 = self.mlp_1(images)
        out_1.register_hook(self.linear_layer_grad_out_hook)
        out_2 = self.linear_layer(out_1)
        out_2.register_hook(self.linear_layer_grad_in_hook)
        out_3 = self.mlp_3(out_2)
        return out_1, out_2, out_3


class TestLinearLayer(unittest.TestCase):
    def test_linear(self):
        """Test functioning of linear layer."""
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
        out_1, out_2, y_hat = model(x)
        # calculate loss
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        # calculate grad
        loss.backward()
        # access grad values
        mapped_grad, mapped_params = access_grad(model)
        # call our linear layer
        # assign values
        input = out_1.detach().numpy()
        output = out_2.detach().numpy()
        grad_in = model.grad_in.detach().numpy()
        grad_out = model.grad_out.detach().numpy()
        grad_gamma = mapped_grad["linear_layer.weight"].detach().numpy().T
        grad_beta = mapped_grad["linear_layer.bias"].detach().numpy()
        weight = mapped_params["linear_layer.weight"].detach().numpy().T
        bias = mapped_params["linear_layer.bias"].detach().numpy()
        custom_linear = Linear(8, 9)
        custom_linear.set_parameters_externally(weight, bias)
        # perform fowrad computation
        custom_out = custom_linear.forward(input)
        decimal_place = 3
        message = "NumPy and reference implementation not almost equal."
        np.testing.assert_array_almost_equal(custom_out, output, decimal_place, message)
        custom_grad_out = custom_linear.backward(grad_in)
        np.testing.assert_array_almost_equal(custom_grad_out, grad_out, decimal_place, message)
        custom_grad_gamma, custom_grad_beta = custom_linear.get_grads()
        np.testing.assert_array_almost_equal(custom_grad_beta, grad_beta, decimal_place, message)


if __name__ == "__main__":
    unittest.main()
