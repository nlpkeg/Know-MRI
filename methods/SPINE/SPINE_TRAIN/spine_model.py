
import torch
from torch import nn
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.INFO)


class SPINEModel(torch.nn.Module):

	def __init__(self, input_dim,hparams):
		super(SPINEModel, self).__init__()
		
		# params
		self.inp_dim = input_dim
		self.hparams = hparams
		self.hdim = self.hparams.hidden_dim*self.inp_dim
		self.noise_level = hparams.noise
		self.getReconstructionLoss = nn.MSELoss()
		self.rho_star = self.hparams.mean_value
		# autoencoder
		self.linear1 = nn.Linear(self.inp_dim, self.hdim)
		self.linear2 = nn.Linear(self.hdim, self.inp_dim)
		

	def forward(self, batch_x, batch_y):
		
		# forward
		batch_size = batch_x.data.shape[0]
		linear1_out = self.linear1(batch_x)
		h = linear1_out.clamp(min=0, max=1) # capped relu
		out = self.linear2(h)

		# different terms of the loss
		reconstruction_loss = self.getReconstructionLoss(out, batch_y) # reconstruction loss
		psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
		asl_loss = self._getASLLoss(h)    	# average sparsity loss
		total_loss = reconstruction_loss + self.hparams.psl*psl_loss + self.hparams.asl*asl_loss
		
		return out, h, total_loss, [reconstruction_loss,psl_loss, asl_loss]


	def _getPSLLoss(self,h, batch_size):
		return torch.sum(h*torch.abs((1-h)))/ (batch_size * self.hdim)


	def _getASLLoss(self, h):
		temp = torch.mean(h, dim=0) - self.rho_star
		temp = temp.clamp(min=0)
		return torch.sum(temp * temp) / self.hdim