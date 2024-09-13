'''This file contains the implementation of the activation functions.


Swish: Activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
HSwish: Activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
HSigmoid: Activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

Credits: osmr on github
'''

from config import nn

class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * nn.sigmoid(x)
    
class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.relu6(x + 3.0, inplace=self.inplace) / 6.0
    
class HSigmoid(nn.Module):
    """
    H-Sigmoid activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.relu6(x + 3.0, inplace=self.inplace) / 6.0