'''This file contains the implementation of the Squeeze-and-Excitation block.

Credits: osmr on github
'''

from config import nn
from helpers import get_activation_layer
from activations import HSigmoid
from conv import conv_builder


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    NOTE: A Feature function should be used prior to this function, such as convolutions.

    Parameters:
    ----------
    channels : int
        Number of channels.
    output_channels : int, default None
        Number of output channels.
    reduction : int, default 8
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 output_channels=None,
                 reduction=8,
                 approx_sigmoid=False,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction
        if not output_channels:
            output_channels = channels

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv_builder(
            in_channels=channels,
            out_channels=mid_channels)
        self.activ_func = get_activation_layer(activation)
        self.conv2 = conv_builder(
            in_channels=mid_channels,
            out_channels=output_channels)
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        # global average pooling
        w = self.pool(x)
        # convolution with activation function
        w = self.conv1(w)
        w = self.activ_func(w)
        # convolution         
        w = self.conv2(w)
        # sigmoid for scaling
        w = self.sigmoid(w)
        x = x * w
        return x