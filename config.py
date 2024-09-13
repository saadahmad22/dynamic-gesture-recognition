'''Imports and configurations for the project'''

from torch import nn
from inspect import isfunction
import torch

dataset_dir = './datasets/HandGesture'
output_dir = './outputs'
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NO", "OK"]
batch_size = 2
num_epochs = 10_000 / 10 # 10 = num of runs per epoch
learning_rate = 0.001
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1474774)

class_labels = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "NO": 10,
    "OK": 11
}