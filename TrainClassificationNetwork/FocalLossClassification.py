import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy

"""
Taken from

https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""


class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=False):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha

		if isinstance(alpha, (float, int)):
			self.alpha = torch.Tensor([alpha, 1-alpha])
		if isinstance(alpha, (list, numpy.ndarray)):
			self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, model_output, target):
		if model_output.dim()>2:
			model_output = model_output.view(model_output.size(0), model_output.size(1), -1)  # N,C,H,W => N,C,H*W
			model_output = model_output.transpose(1, 2)    # N,C,H*W => N,H*W,C
			model_output = model_output.contiguous().view(-1, model_output.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)

		logpt = F.log_softmax(model_output)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type() != model_output.data.type():
				self.alpha = self.alpha.type_as(model_output.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1-pt)**self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()
