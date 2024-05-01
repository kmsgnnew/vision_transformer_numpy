# Vision Transformer - NumPy
This is a NumPy implementation of the [Vision Transformer](https://openreview.net/pdf?id=YicbFdNTTy) and runs on CPU.There is no need of PyTorch. However For benchmarking our implementation with modules of ViT we have used PyTorch in our test folder.

Our goal here, by implementing a replica of ViT using NumPy is to open up the understanding of ViT by showcasing backward propagation along with the forward propagation.

<p align="center">
<img src="images/Vision-Transformer.PNG" width=50% height=50%>
</p>

## Dataset:
For sake of simplicity the code uses MNIST dataset as from [here](http://ldaplusplus.com/files/mnist.tar.gz)

## Training
Both forward and backward propagation is implemented in numpy. The model trained is currently not saved. Loss and metrics are provided.

### Usage:
Python 3.7.0 is used.
1. Install packages using `pip install -r requirements.txt`
2. Update the following arguments in main.py and execute training `python src/main.py`
   * input_path              : Path to the input files
   * batch_size              : Batch size 
   * epochs                  : Total Number of training epochs
   * test_epoch_interval     : Interval for running test set

## Testing Numpy Implementation
Unit tests are available for submodules and can be run using `python -m unittest discover test/`
