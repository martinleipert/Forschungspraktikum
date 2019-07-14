import torch


"""
Martin Leipert
martin.leipert@fau.de

12.07.2019
"""


class FocalLoss(torch.nn.Module):

	def __init__(self, gamma=2, alpha=None):
		self.__gamma = gamma
		self.__alpha = alpha
		self.__eps = 1e-6

	def forward(self, inputdata, target):
		prob = torch.sigmoid(inputdata)
		log_prob = -torch.log(prob)

		focal_loss = torch.sum(torch.pow(1 - prob + self.__eps, self.__gamma).mul(log_prob).mul(target), dim=1)

		if self.__alpha is not None:
			if self.__alpha.type() != inputdata.data.type():
				self.__alpha = self.__alpha.type_as(inputdata.data)
			at = self.__alpha.gather(0, target.data.view(-1))
			focal_loss = focal_loss * torch.Variable(at)

		return focal_loss.mean()
