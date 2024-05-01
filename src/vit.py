import numpy as np
from linear import Linear
from parameter import Parameter
from patch import convert_image_to_patches
from position_embedding import get_positional_embeddings
from vit_block import ViTBlock


class ViT:
    """Vision Transformer"""

    def __init__(self, chw: tuple, n_patches: int, hidden_d: int, n_heads: int, num_blocks: int, out_classses: int):
        """Initialize.

        Args:
            chw: dimension (C H W).
            n_patches: number of patches.
            hidden_d: hidden dimension.
            n_heads: number of heads.
            num_blocks: number of blocks.
            out_classses: total number of output classes.
        """
        self.chw = chw
        self.n_patches = n_patches
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.hidden_d = hidden_d
        self.linear_mapper = Linear(self.input_d, self.hidden_d)
        self.class_token = Parameter(np.random.rand(1, self.hidden_d))
        self.pos_embed = get_positional_embeddings(self.n_patches**2 + 1, self.hidden_d)
        self.blocks = [ViTBlock(hidden_d, n_heads) for _ in range(num_blocks)]
        self.mlp = Linear(self.hidden_d, out_classses)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            images: input array.

        Returns:
            computed linear layer output.
        """
        patches = convert_image_to_patches(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        out = np.stack([np.vstack((self.class_token.val, tokens[i])) for i in range(len(tokens))])
        out = out + self.pos_embed
        for block in self.blocks:
            out = block.forward(out)
        out = self.mlp(out[:, 0])
        return out

    def set_optimizer(self, optimizer_algo: object) -> None:
        """Set optimizer.

        Args:
            optimizer: optimizer.
        """
        self.linear_mapper.set_optimizer(optimizer_algo)
        for block in self.blocks:
            block.set_optimizer(optimizer_algo)
        self.mlp.set_optimizer(optimizer_algo)
        self.class_token.set_optimizer(optimizer_algo)

    def backward(self, error: np.ndarray) -> np.ndarray:
        """Backward propagation.

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error = self.mlp.backward(error)

        for block in self.blocks[::-1]:
            error = block.backward(error)
        removed_cls = error[:, 1:, :]
        _ = self.linear_mapper.backward(removed_cls)
        self.class_token.backward(error[:, 0, :])

    def update_weights(self) -> None:
        """Update weights based on the calculated gradients."""
        self.mlp.update_weights()
        for block in self.blocks[::-1]:
            block.update_weights()
        self.linear_mapper.update_weights()
        self.class_token.update_weights()
