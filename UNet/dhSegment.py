import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np


class dhSegment(nn.Module):

	def __init__(self, num_classes):
		self.__num_classes = num_classes
		self.__input_channels = 3

		self.__pool = nn.MaxPool2d(kernel_size=2, stride=2)
		pass



	# Composed Forward path
	def forward(self, x):
		encoder_outs = []

		# No softmax is used. This means you need to use
		# nn.CrossEntropyLoss is your training script,
		# as this module includes a softmax already.
		x = self.conv_final(x)

		return x

"""
Sub Blocks
"""


class down_convolution_layer(nn.Module):
	def __init__(self, input_dim, subsequent):
		self.__layers = []

		for i in range(subsequent):

			pass

	pass


"""
Helper Methoden
"""


# Method to generate a convolutional layer
# Contracting path
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)

