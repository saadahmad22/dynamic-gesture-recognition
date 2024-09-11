'''This file contains the implementation of the HSigmoid class, which is an approximated sigmoid function.

Credits: osmr on github
'''

from config import nn

class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return nn.relu6(x + 3.0, inplace=True) / 6.0