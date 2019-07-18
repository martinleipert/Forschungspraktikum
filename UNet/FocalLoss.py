"""
Focal loss for segmentaton

Modified and rewritten by
Martin Leipert
martin.leipert@fau.de

Originally by
University of Tokyo Doi Kento 
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py


class FocalLoss2d(nn.Module):

	def __init__(self, gamma=2, weight=None, size_average=True):
		super(FocalLoss2d, self).__init__()

		self.size_average = size_average
		self.gamma = gamma
		if weight is not None:
			self.weight = numpy.float64(weight)
		else:
			self.weight = None

	def forward(self, im_input, target):
		if im_input.dim()>2:
			im_input = im_input.contiguous().view(im_input.size(0), im_input.size(1), -1)
			im_input = im_input.transpose(1,2)
			im_input = im_input.contiguous().view(-1, im_input.size(2)).squeeze()
		if target.dim()==4:
			target = target.contiguous().view(target.size(0), target.size(1), -1)
			target = target.transpose(1,2)
			target = target.contiguous().view(-1, target.size(2)).squeeze()
		elif target.dim()==3:
			target = target.view(-1)
		else:
			target = target.view(-1, 1)

		out = torch.sqrt(im_input.pow(2).sum(1))
		p = im_input / torch.t(out.repeat((4, 1)))

		# compute the negative likelyhood
		ones = torch.ones_like(im_input)
		pt_c = ones - target
		pt = ones + target*p - pt_c*p
		log_pt = torch.log(pt)

		# compute the loss
		loss = -((1-pt)**self.gamma) * log_pt

		loss[torch.isnan(loss)] = 0
		loss[torch.isinf(loss)] = 0

		# Is implemented for four classes
		if self.weight is not None:
			weight = Variable(torch.tensor(self.weight)).to('cuda')
			# Weight the samples accordingly
			tensor_weights = torch.zeros_like(loss)
			repeated_weight = weight.repeat(tensor_weights.size(0), 1)

			tensor_weights = torch.where(target == 1, repeated_weight, tensor_weights)
			loss = tensor_weights * loss

		# averaging (or not) loss
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()
