"""
Focal loss for segmentaton

Modified and rewritten by
Martin Leipert
martin.leipert@fau.de

Originally by
University of Tokyo Doi Kento 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py


class FocalLoss2d(nn.Module):

	def __init__(self, gamma=2, weight=None, size_average=True):
		super(FocalLoss2d, self).__init__()

		self.gamma = gamma
		self.weight = weight

		self.size_average = size_average

	def forward(self, input, target):
		if input.dim()>2:
			input = input.contiguous().view(input.size(0), input.size(1), -1)
			input = input.transpose(1,2)
			input = input.contiguous().view(-1, input.size(2)).squeeze()
		if target.dim()==4:
			target = target.contiguous().view(target.size(0), target.size(1), -1)
			target = target.transpose(1,2)
			target = target.contiguous().view(-1, target.size(2)).squeeze()
		elif target.dim()==3:
			target = target.view(-1)
		else:
			target = target.view(-1, 1)

		out = torch.sqrt(input.pow(2).sum(1))
		p = input / torch.t(out.repeat((4, 1)))

		# compute the negative likelyhood
		ones = torch.ones_like(input)
		pt_c = ones - target

		pt = ones + target*p - pt_c*p

		logpt = torch.log(pt)

		# compute the loss
		loss = -((1-pt)**self.gamma) * logpt

		loss[torch.isnan(loss)] = 0
		loss[torch.isinf(loss)] = 0


		# Is implemented for four classes
		if self.weight:
			weight = Variable(self.weight).to('gpu')
			# Weight the samples accordingly
			tensor_weights = torch.zeros_like(loss)

			for i in range(4):
				tensor_weights = tensor_weights + torch.where(target[:, i, :, :] == 1, weight[i], tensor_weights)
			loss = tensor_weights * loss

		# averaging (or not) loss
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()
