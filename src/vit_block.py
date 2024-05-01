import numpy as np
from gelu import GELU
from layer_normalization import LayerNormalization
from linear import Linear
from multi_head_attention import MultiHeadAttention


class ViTBlock:
    """Vision transformer block. This block is repeated N times in ViT"""

    def __init__(self, hidden_dimension: int, n_heads: int, mlp_ratio: int = 4) -> None:
        """Initialize

        Args:
            hidden_dimension: dimension of feature representation
            n_heads: number of heads
            mlp_ratio: multiplication factor used for determining mlp feature size. Defaults to 4.
        """
        self.hidden_dimension = hidden_dimension
        self.n_heads = n_heads
        self.layer_norm_1 = LayerNormalization(hidden_dimension)
        self.mha = MultiHeadAttention(hidden_dimension, n_heads)
        self.layer_norm_2 = LayerNormalization(hidden_dimension)
        self.mlp_1 = Linear(hidden_dimension, mlp_ratio * hidden_dimension)
        self.gelu = GELU()
        self.mlp_2 = Linear(mlp_ratio * hidden_dimension, hidden_dimension)
        self.cache = dict(input=None)
        self.optimizer = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            x: input array.

        Returns:
            computed softmax output.
        """
        self.cache = dict(input=x)
        mha_out = self.mha.forward(self.layer_norm_1(x))
        stage_1_out = x + mha_out
        out = self.layer_norm_2(stage_1_out)
        out = self.mlp_1(out)
        out = self.gelu(out)
        out = self.mlp_2(out)
        out = out + stage_1_out
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        x = self.cache["input"]
        if grad.shape != x.shape:
            # needed for last block since only cls continues trimming of cls happens outside
            # however since x is accessible this is included here
            grad_in = np.zeros(x.shape)
            grad_in[:, 0] = grad
        else:
            grad_in = grad
        grad_in_skip = grad_in
        grad_in = self.mlp_2.backward(grad_in)
        grad_in = self.gelu.backward(grad_in)
        grad_in = self.mlp_1.backward(grad_in)
        grad_in = self.layer_norm_2.backward(grad_in)
        grad_in = grad_in + grad_in_skip
        grad_in_skip = grad_in
        grad_in = self.mha.backward(grad_in)
        grad_in = self.layer_norm_1.backward(grad_in)
        grad_in = grad_in + grad_in_skip
        return grad_in

    def set_optimizer(self, optimizer: object) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.layer_norm_1.set_optimizer(optimizer)
        self.layer_norm_2.set_optimizer(optimizer)
        self.mha.set_optimizer(optimizer)
        self.mlp_1.set_optimizer(optimizer)
        self.mlp_2.set_optimizer(optimizer)

    def update_weights(self) -> None:
        """Update weights."""
        self.layer_norm_1.update_weights()
        self.layer_norm_2.update_weights()
        self.mha.update_weights()
        self.mlp_1.update_weights()
        self.mlp_2.update_weights()
