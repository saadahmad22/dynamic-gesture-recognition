'''Defines various convolutions

Credits: osmr on github
'''

from config import nn

def conv_builder(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple=1,
    stride: int | tuple=1,
    padding: int | tuple | str=0,
    padding_mode: str='zeros',
    dilation: int | tuple=1,
    groups=1,
    bias=True) -> nn.Conv2d:
    """Create a convolution layer.
    
    For the parameters, in case a tuple is taken the first value is for the height and the second for the width.

    When groups == in_channels and out_channels == K * in_channels, 
    where K is a positive integer, this operation is also known as a “depthwise convolution”

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Convolution kernel size.
    stride : int or tuple of int
        Stride of the convolution.
    padding : int or tuple of int
        Padding value for convolution layer.
        - If int: uses the same padding in all directions.
        - If tuple: provides the padding separately for the height and width.
        - If str: 
            "valid" (i.e., 0), 
            "same" (pads the input so the output has the shape as the input).
                - This mode doesn't support any stride values other than 1
    padding_mode : str
        Padding mode for convolution layer.
        - 'zeros', 'reflect', 'replicate' or 'circular'.
    dilation : int or tuple of int
        Dilation value for convolution layer.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        f True, adds a learnable bias to the output
    """

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias)