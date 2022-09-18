from decay_rnn import DECAY_RNN 
from slacked import SDRNN
from ablation import ABDRNN
from ordered_neuron import ONLSTM
import torch
import sys
import torch.nn as nn

loss_func  =  nn.CrossEntropyLoss()

# a new self attention mechanims, suggested by Maini et. al.(2020)
class MAXATTN(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MAXATTN, self).__init__()
        self.attention_layer =  nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, hidden, key=None, value=None):
        # hidden (T, bsz, dim) 
        # in this case all three are same!!
        T =  hidden.size(0)
        query =  torch.max(hidden, dim=0, keepdim=True)[0] #(1, bsz, dim)
#         print(query.shape)
        out, weight = self.attention_layer(query, hidden, hidden)
        # just to keep consistent with what we are doing so far... return (T, bsz, dim) and (bsz, T, T) 
        return torch.cat([out for i in range(T)], dim=0), torch.cat([weight for i in range(T)], dim=1) 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attn_mask(L, S):
	return torch.triu(-1*torch.tensor([float("Inf") for i in range(L*S)]).view(L, S), diagonal=1).to(device)

class Agreement_model(nn.Module):

	def __init__(self, rnn_arch, embedding_size, hidden_size, vocab_size, num_layers, output_size, dropout=0., emb_attention=False, act_attention=False, max_attn=False, num_heads=1, activation='relu'):
		super(Agreement_model, self).__init__()
		self.rnn_arch = rnn_arch
		self.embedding_size = embedding_size
		self.hidden_size=hidden_size
		self.vocab_size = vocab_size
		self.activation=activation
		self.num_layers = num_layers
		self.output_size = output_size
		self.dropout = dropout
		self.num_heads=num_heads
		self.emb_attention=emb_attention
		self.act_attention=act_attention
		self.max_attn = max_attn

		if emb_attention:
			self.attention_layer = nn.MultiheadAttention(self.embedding_size,self.num_heads)
		elif act_attention:
			self.attention_layer = nn.MultiheadAttention(self.hidden_size,self.num_heads)			
		if max_attn and not emb_attention:
			self.attention_layer = MAXATTN(self.hidden_size, self.num_heads)
		elif max_attn and emb_attention:
			raise NotImplementedError("To be implemented")

		self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size)
		if self.rnn_arch == 'LSTM':
			self.recurrent_layer = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout)
		elif self.rnn_arch == "GRU" :
			self.recurrent_layer = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout)
		elif self.rnn_arch == "RNN":
			self.recurrent_layer = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout)
		elif self.rnn_arch == "DECAY":	
			self.recurrent_layer = DECAY_RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout, activation=self.activation)
		elif self.rnn_arch == 'ONLSTM':
			self.recurrent_layer = ONLSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=self.dropout)
		elif self.rnn_arch == 'SDRNN':
			self.recurrent_layer = SDRNN(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout, activation=self.activation)
		elif self.rnn_arch=='ABDRNN':
			self.recurrent_layer = ABDRNN(self.embedding_size, self.hidden_size, self.num_layers, dropout= self.dropout, activation=self.activation)
		else:
			print("Invalid architecture request!!! Exiting")
			sys.exit()
		self.output_layer = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input_, hidden=None):
		attention_weights=None 
		# we need to assume that the input is of form : (bsz, time_stamps) 
		# then we need to permute the axis and then pass it to the recurrent layers 
		input_  = self.embedding_layer(input_)  # bsz, time_stamps, embedding_size 
		input_  = input_.permute(1, 0, 2)
		if self.emb_attention:
			input_ , attention_weights = self.attention_layer(input_, input_, input_, attn_mask=get_attn_mask(input_.size(0),input_.size(0)))
		if self.rnn_arch== 'LSTM' or self.rnn_arch =='ONLSTM':
			output, (h_n, c_n) = self.recurrent_layer(input_, hidden) #output : (time_stamps, bsz, hidden_size) , h_n: (num_layers, bsz, hidden_size)
		else:
			output, h_n = self.recurrent_layer(input_, hidden) #output : (time_stamps, bsz, hidden_size) , h_n: (num_layers, bsz, hidden_size)
		if self.act_attention or self.max_attn:
			output, attention_weights = self.attention_layer(output, output, output)  # applying SA to all the hidden units. 

		h_last = output.permute(1, 0, 2) # (bsz, time_stamps, hidden_size)
		output =  self.output_layer(h_last) #(bsz, time_stamps, output_size)

		if  self.rnn_arch== 'LSTM' or self.rnn_arch =='ONLSTM':
			return output, h_last, (h_n, c_n), attention_weights
		else: 
			return output, h_last, h_n,  attention_weights    #(bsz, time_stamps, output_size), (bsz, time_stamps, hidden_size) , (num_layers, bsz, hidden_size)


	def predict(self, input_, ground_truth, hidden=None, compute_loss=False):

		attention_weights=None
		input_bsz = input_.size(0)
		if  self.rnn_arch== 'LSTM' or self.rnn_arch =='ONLSTM':
			output, h_last, (h_n, c_n), attention_weights = self.forward(input_, hidden) # output is (bsz, time_stamps, hidden_size) and we need to take final time stamp value 
		else:
			output, h_last, h_n, attention_weights = self.forward(input_, hidden)

		ground_truth = ground_truth.reshape(input_bsz,)
		prediction = torch.argmax(output[:, -1, :], dim= -1).view(input_bsz, 1)

		acc  = torch.sum(torch.eq(prediction, ground_truth.view(input_bsz, 1)))  # this accuracy is just the number of correct examples!!!  WATCH OUT!

		# for indices of data with wrong predictions
		incorr_indices = []
		for index, pred in enumerate(prediction):
			if prediction[index]!=ground_truth.view(input_bsz, 1)[index]:
				incorr_indices.append(index)


		if compute_loss:
			loss = loss_func(output[:, -1, :], ground_truth.long())
			if self.act_attention or self.max_attn:
				return loss, output, prediction, acc, attention_weights[:, -1, :], h_n, h_last
			else:
				return loss, incorr_indices, output, prediction, acc, h_n, h_last
		else:
			if self.act_attention or self.max_attn:
				return output, prediction, acc, attention_weights[:, -1, :], h_n, h_last
			else:
				return output, prediction, acc, h_n, h_last
