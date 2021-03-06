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
from torch.autograd import Variable

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py


class FocalLoss2d(nn.Module):

	def __init__(self, gamma=2, weight=None, size_average=True):
		super(FocalLoss2d, self).__init__()

		self.size_average = size_average
		self.gamma = numpy.float64(gamma)
		if weight is not None:
			self.weight = torch.from_numpy(numpy.float64(weight))
			self.weight = self.weight.to("cuda")
		else:
			self.weight = None

	def forward(self, im_input, target):
		im_input = im_input.double()
		target = target.double()

		if im_input.dim() > 2:
			im_input = im_input.contiguous().view(im_input.size(0), im_input.size(1), -1)
			im_input = im_input.transpose(1,2)
			im_input = im_input.contiguous().view(-1, im_input.size(2)).squeeze()
		if target.dim() == 4:
			target = target.contiguous().view(target.size(0), target.size(1), -1)
			target = target.transpose(1,2)
			target = target.contiguous().view(-1, target.size(2)).squeeze()
		elif target.dim() == 3:
			target = target.view(-1)
		else:
			target = target.view(-1, 1)

		# Normalized probabilities
		out = torch.sqrt(im_input.pow(2).sum(1))
		p = im_input / torch.t(out.repeat((4, 1)))

		# compute the negative likelyhood
		ones = torch.ones_like(im_input)
		pt_c = ones - target

		pt = target*p + pt_c*(1-p)
		log_pt = torch.log(pt)

		# compute the loss
		loss = -((1.0-pt).pow(self.gamma)) * log_pt

		# Is implemented for four classes
		if self.weight is not None:
			weight = self.weight
			# Weight the samples accordingly
			tensor_weights = torch.zeros_like(loss)
			repeated_weight = weight.repeat(tensor_weights.size(0), 1)

			tensor_weights = torch.where(target == 1, repeated_weight, tensor_weights)
			loss = tensor_weights * loss

		loss[torch.isnan(loss)].detach()
		loss[torch.isnan(loss)] = 0
		loss[torch.isinf(loss)].detach()
		loss[torch.isinf(loss)] = 0

		loss = loss.sum(dim=1)

		# averaging (or not) loss
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()
