import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
from numpy.random import binomial
import sys

_VF = torch._C._VariableFunctions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rectify(x):
	relu = nn.ReLU()
	return relu(x)

class DecayModule(nn.Module):


	def __init__(self, input_size , hidden_size, bias = True, num_chunks = 1, activation='relu', nodiag=False):
		super(DecayModule, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.nodiag=nodiag
		self.bias = bias
		self.num_chunks = num_chunks
		self.rgate = nn.Parameter(torch.tensor(0.8),requires_grad=True)
		self.weight_ih = nn.Parameter(torch.Tensor(num_chunks*hidden_size,input_size))
		self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
		self.d_rec = nn.Parameter(torch.zeros(num_chunks * hidden_size, hidden_size),requires_grad=False)
		self.activation=activation

		if bias:
			self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
			self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
		else:
			self.register_parameter('bias_ih', None)
			self.register_parameter('bias_hh', None)
		
		self.reset_parameters()
		if self.nodiag:
			for i in range(hidden_size):
				self.weight_hh.data[i, i]=0

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			nn.init.uniform_(weight, -stdv, stdv)
		
		for name,param in self.named_parameters():
			if name=="rgate":
				param.data  = torch.tensor(0.8)

		for i in range(self.num_chunks) :
			x = i * self.hidden_size
			for j in range(self.hidden_size) :
				if (j < 0.8*self.hidden_size) :
					self.d_rec[x + j][j] = 1.0
				else :
					self.d_rec[x + j][j] = -1.0

	def forward(self, input_, hx = None):
		if hx is None:
			hx = input_.new_zeros(self.num_chunks*self.hidden_size, requires_grad=False)

		# dale_hh = torch.mm(self.relu(self.weight_hh), self.d_rec)

		if (self.bias) :
			w_x = self.bias_ih + torch.matmul(self.weight_ih,input_).t()
			# w_h = self.bias_hh + torch.matmul(dale_hh,hx.t()).t()
			# w_h = self.bias_hh + torch.matmul(self.weight_hh,hx.t()).t()  
		else :
			w_x = torch.matmul(self.weight_ih,input_).t()
			# w_h = torch.matmul(dale_hh,hx.t()).t()
			# w_h = torch.matmul(self.weight_hh,hx.t()).t()  
		w_w = ((self.rgate) * hx) + ((1-(self.rgate)) * (w_x))
		if self.activation=='tanh':
			h = self.tanh(w_w)
		else:
			h = self.relu(w_w)
		return h  # shape is (M, H) where M is the batch size and H is the embedded dimension. 

class ABDRNN(nn.Module):


	def __init__(self, input_size, hidden_size, num_layers = 1, dropout=0.2, activation='relu', nodiag=False):
		super(ABDRNN, self).__init__()

		self.input_size  = input_size 
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.nodiag=nodiag
		self.dropout = dropout
		self.dropout_layer = nn.Dropout(dropout)
		self.activation=activation

		for layer in range(num_layers):
			layer_input_size  = self.input_size if layer == 0 else self.hidden_size
			cell = DecayModule(input_size  = layer_input_size, hidden_size = self.hidden_size,activation=self.activation, nodiag=self.nodiag)
			setattr(self, 'cell_{}'.format(layer), cell)

		self.reset_parameters()

	def get_cell(self, layer):
		return getattr(self, 'cell_{}'.format(layer))

	def reset_parameters(self):
		for layer in range(self.num_layers):
			cell = self.get_cell(layer)
			cell.reset_parameters()

	def forward_single_layer(self, input_, layer, h_0=None, max_time=50):
		# assumed input shape : (time_stamps, batch_size, input_emb_dim)
		# assumed output shape: (time_stamps, batch_size, hidden_dim) , h_n shape = (num_layers, batch_size, hidden_dim) --> corresponding to the last time stamp 
		all_hidden = []
		cell = self.get_cell(layer)
		max_time, M = input_.shape[0], input_.shape[1]
		state = h_0

		assert (h_0.shape[0],h_0.shape[1]) == (M, self.hidden_size)

		for time in range(max_time):
			state = cell(input_ = input_[time, :, :].t(), hx = state)
			if layer==self.num_layers-1:
				all_hidden.append(state)
			else:
				all_hidden.append(self.dropout_layer(state))

		h_n = state #last time stamp state (M, H)
		all_hidden = torch.stack(all_hidden)
		assert (h_n.shape[0], h_n.shape[1]) ==(M, self.hidden_size)
		return all_hidden, h_n

	def forward(self, input_, h_0=None, max_time=50):
		# we assume that the input shape is (time_stamps, batch_size, input_sizes)
		# for every example the h_init will serve as none. H_init will be none in each layer and for all examples. 
		# the inputs that will be passed to layer_0 will be the input_, for the subsequent layers, we will pass the processed 
		# hidden layer outputs. 
		# h_0 is the inital state to be used for the dynamics. 

		max_time, M = input_.shape[0], input_.shape[1]

		if not torch.is_tensor(h_0):
			h_0 = torch.zeros((self.num_layers, M, self.hidden_size)).to(device)

		h_n = []
   
		for layer in range(self.num_layers):
			if layer == 0: 
				all_hidden, h_n_layer = self.forward_single_layer(input_, layer, h_0[layer, :, :])
			else:
				all_hidden, h_n_layer = self.forward_single_layer(all_hidden, layer, h_0[layer,:,:])

			h_n.append(h_n_layer)

		h_n = torch.stack(h_n)

		assert (h_n.shape[0],h_n.shape[1],h_n.shape[2]) ==( self.num_layers, M , self.hidden_size )
		assert (all_hidden.shape[0],all_hidden.shape[1],all_hidden.shape[2])==(max_time, M, self.hidden_size)

		return all_hidden, h_n