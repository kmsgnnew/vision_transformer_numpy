import sys
import unittest

sys.path.append(".")
import numpy as np
from einops import rearrange
from src.patch import convert_image_to_patches


def ref_convert_image_to_patches(images, patch_height, patch_width):
    """Convert Images to patches reference implementation using einops.

    Args:
        images: input images
        patch_height: patch_height
        patch_width: patch width

    Returns:

    """
    out_patches = (rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_height, p2=patch_width),)
    return out_patches


class TestPatch(unittest.TestCase):
    def test_patching(self) -> None:
        """Test that it does patching."""
        images_np = np.random.rand(8, 3, 28, 28)  # N C H W
        out_np = convert_image_to_patches(images_np, 7)
        out_ref = ref_convert_image_to_patches(images_np, 4, 4)
        decimal_place = 3
        message = "NumPy and reference not almost equal."
        np.testing.assert_array_almost_equal(out_np, out_ref[0], decimal_place, message)


if __name__ == "__main__":
    unittest.main()
