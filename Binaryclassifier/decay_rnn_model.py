import os
import sys
import os.path as op
import random
import torch
import torch.nn as nn
import six
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import filenames
from fullgram_model import Agreement_model
import POS_Tagger
from utils import deps_from_tsv, dump_template_waveforms
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
	if not hasattr(sequences, '__len__'):
		raise ValueError('`sequences` must be iterable.')
	lengths = []
	for x in sequences:
		if not hasattr(x, '__len__'):
			raise ValueError('`sequences` must be a list of iterables. '
							 'Found non-iterable: ' + str(x))
		lengths.append(len(x))

	num_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)

	# take the sample shape from the first non empty sequence
	# checking for consistency in the main loop below.
	sample_shape = tuple()
	for s in sequences:
		if len(s) > 0:
			sample_shape = np.asarray(s).shape[1:]
			break

	is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
	if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
		raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
						 "You should set `dtype=object` for variable length strings."
						 .format(dtype, type(value)))

	x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
	for idx, s in enumerate(sequences):
		if not len(s):
			continue  # empty list/array was found
		if truncating == 'pre':
			trunc = s[-maxlen:]
		elif truncating == 'post':
			trunc = s[:maxlen]
		else:
			raise ValueError('Truncating type "%s" '
							 'not understood' % truncating)

		# check `trunc` has expected shape
		trunc = np.asarray(trunc, dtype=dtype)
		if trunc.shape[1:] != sample_shape:
			raise ValueError('Shape of sample %s of sequence at position %s '
							 'is different from expected shape %s' %
							 (trunc.shape[1:], idx, sample_shape))

		if padding == 'post':
			x[idx, :len(trunc)] = trunc
		elif padding == 'pre':
			x[idx, -len(trunc):] = trunc
		else:
			raise ValueError('Padding type "%s" not understood' % padding)
	return x


class BatchedDataset(Dataset):
	'''
	This class make a general dataset that we will use to generate 
	the batched training data
	'''
	def __init__(self, x_train, y_train, verb_loc):
		super(BatchedDataset, self).__init__()
		self.x_train = x_train
		self.y_train = y_train
		self.verb_loc = verb_loc
		assert (x_train).shape[0] == (y_train).shape[0] 
		self.length =  (x_train).shape[0]
	def __getitem__(self, index):
		return self.x_train[index], self.y_train[index], self.verb_loc[index]

	def __len__(self):
		return self.length


class ood_dataset(Dataset):
	'''
	This class make a general dataset that we will use to generate 
	the batched training data
	'''
	def __init__(self, x_sent, x_train, y_train):
		super(ood_dataset, self).__init__()
		self.x_sent = x_sent
		self.x_train = x_train
		self.y_train = y_train
		assert (x_train).shape[0] == (y_train).shape[0] 
		self.length =  (x_train).shape[0]
	def __getitem__(self, index):
		return self.x_sent[index], self.x_train[index], self.y_train[index]

	def __len__(self):
		return self.length	

class Demarcated_dataset(Dataset):
	'''
	This class make a general dataset that we will use to generate 
	the batched training data
	'''
	def __init__(self, x_train, y_train):
		super(Demarcated_dataset, self).__init__()
		self.x_train = x_train
		self.y_train = y_train
		assert (x_train).shape[0] == (y_train).shape[0] 
		self.length =  (x_train).shape[0]
	def __getitem__(self, index):
		return self.x_train[index], self.y_train[index]
	def __len__(self):
		return self.length


class DECAY_RNN_Model(object):

	serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
						 'X_train', 'Y_train', 'deps_train',
						 'X_test', 'Y_test', 'deps_test']

	def log(self, message):
		with open('logs/' + self.output_filename, 'a') as file:
			file.write(str(message) + '\n')

	def log_demarcate_train(self, message):
		with open('logs/demarcated_train_acc_' + self.output_filename, 'a') as file:
			file.write(str(message) + '\n')

	def log_demarcate_val(self, message):
		with open('logs/demarcated_val_acc_' + self.output_filename, 'a') as file:
			file.write(str(message) + '\n')

	def log_val(self, message):
		with open('logs/val_' + self.output_filename, 'a') as file:
			file.write(str(message) + '\n')
	def log_grad(self, message):
		with open('logs/grad_' + self.output_filename, 'a') as file:
			file.write(message + '\n')

	def log_alpha(self,message):
		with open('logs/alpha_' + self.output_filename, 'a') as file:
			file.write(message + '\n')

	def __init__(self, rnn_arch, filename=None, embedding_size=50, hidden_size = 50, output_size=10, num_layers=1, dropout=0.2,  
				 maxlen=50, prop_train=0.9, mode='infreq_pos', vocab_file=filenames.vocab_file,
				 equalize_classes=False, criterion=None, len_after_verb=0,
				 output_filename='default.txt'):
		'''
		filename: TSV file with positive examples, or None if unserializing
		criterion: dependencies that don't meet this criterion are excluded
			(set to None to keep all dependencies)
		verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
		'''
		self.rnn_arch = rnn_arch
		self.filename = filename
		self.num_layers=num_layers
		self.dropout=dropout
		self.vocab_file = vocab_file
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.prop_train = prop_train
		self.mode = mode
		self.maxlen = maxlen
		self.equalize_classes = equalize_classes
		self.criterion = (lambda x: True) if criterion is None else criterion
		self.len_after_verb = len_after_verb
		self.output_filename = output_filename

	def create_train_test_dataloader(self):
		x_train = torch.tensor(self.X_train, dtype=torch.long).to(device)
		y_train = torch.tensor(self.Y_train).to(device)
		self.deps_train = list(self.deps_train)
		verb_loc_train = torch.stack([torch.tensor(elem["verb_index"]) for elem in self.deps_train]).to(device).squeeze()
		self.Train_DataGenerator =  DataLoader(BatchedDataset(x_train, y_train, verb_loc_train), batch_size= self.train_bsz, shuffle=True, drop_last=False)

		x_test = torch.tensor(self.X_test, dtype=torch.long).to(device)
		y_test = torch.tensor(self.Y_test).to(device)
		self.deps_test = list(self.deps_test)
		verb_loc_test = torch.stack([torch.tensor(elem["verb_index"]) for elem in self.deps_test]).to(device).squeeze()
		self.Test_DataGenerator =  DataLoader(BatchedDataset(x_test, y_test, verb_loc_test), batch_size= self.test_bsz, shuffle=True, drop_last=False)    

	def create_train_test_dataloader_POS(self):
		x_train = torch.tensor(self.X_train, dtype=torch.long).to(device)
		pos_train = torch.tensor(self.POS_train).to(device)
		self.deps_train = list(self.deps_train)
		verb_loc_train = torch.stack([torch.tensor(elem["verb_index"]) for elem in self.deps_train]).to(device).squeeze()
		self.Train_DataGenerator =  DataLoader(BatchedDataset(x_train, pos_train, verb_loc_train), batch_size= self.train_bsz, shuffle=True, drop_last=False)

		x_test = torch.tensor(self.X_test, dtype=torch.long).to(device)
		pos_test = torch.tensor(self.POS_test).to(device)
		self.deps_test = list(self.deps_test)
		verb_loc_test = torch.stack([torch.tensor(elem["verb_index"]) for elem in self.deps_test]).to(device).squeeze()
		self.Test_DataGenerator =  DataLoader(BatchedDataset(x_test, pos_test, verb_loc_test), batch_size= self.test_bsz, shuffle=True, drop_last=False)   

	def create_demarcated_dataloader(self):
		training_demarcated_dataloader={}
		testing_demarcated_dataloader={}
		if hasattr(self, 'testing_dict'):
			for key in self.testing_dict.keys():
				x_test, y_test = zip(*self.testing_dict[key])
				x_test=torch.tensor(list(x_test), dtype=torch.long).to(device)
				y_test=torch.tensor(list(y_test)).to(device)
				testing_demarcated_dataloader[key] = DataLoader(Demarcated_dataset(x_test, y_test), batch_size= self.test_bsz, shuffle=True, drop_last=False)
		
			self.testing_demarcated_dataloader = testing_demarcated_dataloader

		if hasattr(self, 'training_dict'):
			for key in self.training_dict.keys():
				x_train, y_train = zip(*self.training_dict[key])
				x_train=torch.tensor(list(x_train), dtype=torch.long).to(device)
				y_train=torch.tensor(list(y_train)).to(device)
				training_demarcated_dataloader[key] = DataLoader(Demarcated_dataset(x_train, y_train), batch_size= self.test_bsz, shuffle=True, drop_last=False)
		
			self.training_demarcated_dataloader = training_demarcated_dataloader


	def create_OOD_dataloader(self, final_dict_testing):
		# for every key, create a dict of dataloaders...
		ood_dataloader_dict={}
		for key in final_dict_testing.keys():
			x_sent = final_dict_testing[key][0]
			x =  torch.tensor(final_dict_testing[key][1], dtype=torch.long).to(device)
			y =  torch.tensor(final_dict_testing[key][2]).to(device)
			if len(x)==0: continue
			ood_dataloader_dict[key]=DataLoader(ood_dataset(x_sent, x, y), batch_size= self.train_bsz, shuffle=True, drop_last=False)
		return ood_dataloader_dict

	def demark_testing(self):
		X_test=self.X_test
		Y_test=self.Y_test
		deps_test=self.deps_test
		print("Size of X_test is {}".format(len(X_test)))
		testing_dict={}
		assert len(X_test)==len(Y_test) and len(Y_test)==len(deps_test)
		for i in (range(len(X_test))):
			key = (deps_test[i]['n_intervening'], deps_test[i]["n_diff_intervening"])
			if not key in testing_dict.keys():
				testing_dict[key]=[]
			testing_dict[key].append((X_test[i], Y_test[i]))
		self.testing_dict=testing_dict

		X_train =self.X_train
		Y_train =self.Y_train
		deps_train=self.deps_train
		print("Size of X_train is {}".format(len(X_train)))
		training_dict={}
		assert len(X_train)==len(Y_train) and len(Y_train)==len(deps_train)
		for i in (range(len(X_train))):
			key = (deps_train[i]['n_intervening'], deps_train[i]["n_diff_intervening"])
			if not key in training_dict.keys():
				training_dict[key]=[]
			training_dict[key].append((X_train[i], Y_train[i]))
		self.training_dict=training_dict


	def result_demarcated(self):
		if not hasattr(self, "testing_dict") or not hasattr(self, 'training_dict'):
			print('creating demarcated dict!')
			self.demark_testing()
			self.create_demarcated_dataloader()

		result_dict={}
		self.model.eval()
		if not self.demarcate_train:
			loader_dict= self.testing_demarcated_dataloader
		else:
			loader_dict= self.training_demarcated_dataloader
		hidden_state_dict={}
		with torch.no_grad():
			for key in loader_dict.keys():
				loader = loader_dict[key]
				correct=0
				example_processed=0
				hidden_state_dict[key]=[]
				for x_test, y_test in loader:
					bsz = x_test.size(0)
					x_test = x_test.view(bsz, self.maxlen)
					y_test = y_test.view(bsz, )
					example_processed+=bsz
					if self.act_attention or self.max_attn:
						_, pred, _, acc, attention_weights, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 
					else:
						_, pred, _, acc, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 					
					correct+=acc
					hidden_state_dict[key].append((h_n, pred, y_test))

				result_dict[key] = (float(correct)/float(example_processed), float(example_processed))

		correct=0
		total=0
		for key in result_dict.keys():
			correct+= result_dict[key][0]*result_dict[key][1]
			total+=result_dict[key][1]
		acc = float(correct)/float(total)

		# self.log(str(result_dict))
		self.log(str(acc))        
		self.model.train()

		# with open('hidden_dict.pkl', 'wb') as f :
		# 	pickle.dump(hidden_state_dict, f)
			

		return acc, result_dict

	def create_model(self):
		self.log('Creating model')
		self.log('vocab size : ' + str(len(self.vocab_to_ints)))
		print('Creating model')
		print('vocab size : ' + str(len(self.vocab_to_ints)))
		if self.embedding_size%self.nheads !=0:
			print("Num heads should divide embedding size. Exiting !!!")
			sys.exit()
		if not self.train_tagger:
			self.model = Agreement_model(rnn_arch= self.rnn_arch, embedding_size = self.embedding_size, hidden_size=self.hidden_size, vocab_size = len(self.vocab_to_ints)+1, num_layers=self.num_layers, output_size=self.output_size, dropout=self.dropout, act_attention=self.act_attention, max_attn=self.max_attn, num_heads=self.nheads, activation=self.activation)
		else:
			self.model = POS_Tagger.Agreement_model(rnn_arch= self.rnn_arch, embedding_size = self.embedding_size, hidden_size=self.hidden_size, vocab_size = len(self.vocab_to_ints)+1, num_layers=self.num_layers, output_size=self.output_size, dropout=self.dropout, act_attention=self.act_attention, max_attn=self.max_attn, num_heads=self.nheads, activation=self.activation)
		self.model = self.model.to(device)

##########################################################
#### EXTERNAL ADDITION DONE TO GET LINZEN TESTED #########
##########################################################
	def test_ood(self, final_dict_testing):
		final_dict_testing =  self.create_OOD_dataloader(final_dict_testing)
		result={}
		hidden_state_dict={}
		with torch.no_grad():
			self.model.eval()
			for key in final_dict_testing:
				loss_=0
				correct=0
				hidden_last=[]
				y_out=[]
				y_pred=[]
				total = 0
				for x_sent, x_test, y_test in final_dict_testing[key]:
					bsz = x_test.size(0)
					x_test = x_test.view(bsz, self.maxlen)
					y_test = y_test.view(bsz, )
					if self.act_attention or self.max_attn:
						loss, pred, _, acc, _, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 
					else:
						loss, pred, _, acc, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 					
					loss_+=loss*bsz
					correct+=acc
					total+=bsz
					hidden_last.append((h_n[-1]))
					y_out.append(y_test)
					y_pred.append(pred)

				result[key] =  (float(correct)/float(total), total)
				hidden_state_dict[key]  = (hidden_last, y_out, y_pred)

		self.log(result)
		print(result)
		for key in hidden_state_dict.keys():
			with open(str(key)+'.pkl', 'wb') as f:
				pickle.dump(hidden_state_dict[key], f)

	def ood_genset_creation(self, filename, save_processed_data = False):
		value = lambda key : int(key.split("_")[0]!="sing") # only requried in PVN
		ex_list = []    #specialized method for waveforms visualizations 
		for files in os.listdir(filename):
			loaded_template = pickle.load(open(os.path.join(filename, files), 'rb'))
			ex_list.append((loaded_template, files))
		test_example={}
		for i in range(len(ex_list)):
			for keys in ex_list[i][0].keys():
				list1= ex_list[i][0][keys]
				if len(list1[0]) > 2:     #ignoring the 3 tuples in the templates 
					continue
				if (ex_list[i][1], keys) in test_example.keys():    # this just means if (file, singular) is a key to test_example dictionary, if not then initialize the key 
					pass
				else:
					test_example[(ex_list[i][1], keys)]=[]
				for X in list1:            
					# x, _ = X   
					# # x = correct sentence, x_neg is ungrammatical sentence due to inflection (but for pvn we need to focus only on grammatical sentence)
					# # note that for pvn task, where we want to make the waveforms, here the label will be 0 for singulars and 1 for plural 
					# test_example[(ex_list[i][1], keys)].append((x, value(keys)))   # we need to by pass this step for y waveform construction
					x, x_neg =  X
					test_example[(ex_list[i][1], keys)].append((x, 0))
					test_example[(ex_list[i][1], keys)].append((x_neg, 1))

		external_testing_dict={}
		for keys in test_example.keys():
			x_test_, y_test_ = zip(*test_example[keys])
			external_testing_dict[keys] = (x_test_, y_test_)
		
	# At this time we have a dictionary that has key -->(filename, property) and value a tuple  (X_test(string form), y_test)

		final_dict_testing = self.valid_input(external_testing_dict)

		if save_processed_data:
			os.mkdir("Testing_data")
			for keys in final_dict_testing.keys():
				pickle_out = open(os.path.join("Testing_data", str(keys))+".pkl", "wb")     
				pickle.dump(final_dict_testing[keys], pickle_out)

		self.test_ood(final_dict_testing)

	def valid_input(self,  external_testing_dict):
		final_dict_testing={}
		for keys in external_testing_dict.keys():
			x = []
			y = []
			x_sentences = []
			X_test, Y_test = external_testing_dict[keys]
			for i in range(len(X_test)):
				x_ex = []
				flag=True
				example = X_test[i]
				token_list = example.split()
				if len(token_list)>self.maxlen:   #ignore big sentences than max len 
					continue
				for tokens in token_list:
					if not tokens in self.vocab_to_ints.keys():   #if unknown character, leave the example 
						flag=False
						break
					x_ex.append(self.vocab_to_ints[tokens])
				if not flag:
					continue
				x.append(x_ex)
				x_sentences.append(X_test[i])
				y.append(Y_test[i])
				
			x = pad_sequences(x, self.maxlen)
			final_dict_testing[keys]=(x_sentences, x, y)
			assert len(x_sentences) == len(x) == len(y), "assert failed! length of sentences is different from length of actual testing sentence per template per sing/plur"
		return final_dict_testing

	def pipeline(self, train, train_bsz =128, test_bsz=32,load = False, model = '', test_size=7000, model_prefix='_', num_epochs=20, load_data=False, save_data=False, test_linzen_template_pvn=False, linzen_template_filename=None, 
				train_size=None, data_name='Not', lr=0.001, annealing=False, nheads=1, activation='relu', act_attention=False, max_attn=False, use_hidden=False, K=5, L=1, train_tagger=False, compare_models=False, m1=None, m2=None, domain_adaption = False, 
				test_demarcated=False, demarcate_train=False, ood=False, augment_train=True, augment_test=True, augment_ratio=1, verb_embedding=False):

		self.train_bsz = train_bsz
		self.test_bsz = test_bsz
		self.train=train
		self.test_size = test_size
		self.num_epochs = num_epochs
		self.test_linzen_template_pvn = test_linzen_template_pvn
		self.linzen_template_filename =linzen_template_filename
		self.model_name = model
		self.model_prefix = model_prefix
		self.lr = lr
		self.annealing=annealing
		self.nheads= nheads
		self.activation=activation
		self.act_attention=act_attention
		self.use_hidden=use_hidden
		self.max_attn = max_attn
		self.K = K
		self.L = L
		self.train_tagger=train_tagger
		self.train=train
		self.domain_adaption = domain_adaption
		self.test_demarcated=test_demarcated
		self.demarcate_train=demarcate_train
		self.ood = ood
		self.augment_train=augment_train
		self.augment_test=augment_test
		self.augment_ratio=augment_ratio
		self.verb_embedding	=verb_embedding	

		if load_data:
			self.load_train_and_test(test_size, data_name, self.domain_adaption)
			print("Loading Data!")
		else :
			self.log('creating data')
			print("Creating data")
			examples = self.load_examples(data_name, save_data, None if train_size is None else train_size*10)
			self.create_train_and_test(examples, test_size, data_name, save_data)
			# at this time we have the train and testing data in our hand !! 
		
		if train_tagger:
			self.create_train_test_dataloader_POS()
		else:	
			self.create_train_test_dataloader()

		if load :
			if compare_models:
				pass
			else:
				self.load_model()
		else:
			self.create_model()

		if self.train_tagger:
			self.train_tagging()
		elif self.train:
			self.train_model()
			print("Training Done!!! Now evaluating the full model!")
			self.test_size=0
			self.load_train_and_test(test_size, data_name, self.domain_adaption)
			self.load_model()
			self.create_train_test_dataloader()
			acc = self.validate()
			print("Testing complete!!!")
			print(acc)
			self.log(acc)
			# print("Following is average attention weight distribution for the testing examples! 0 key means verb loc")
			# print(self.attn_dist) 
			self.log(acc)
		elif compare_models:
			m1 = torch.load(m1)
			m2 = torch.load(m2)
			self.compare_models(m1, m2)
		elif self.verb_embedding :
			self.verb_emb()
		else:
			if self.test_demarcated:
				acc, result_dict = self.result_demarcated()
				print("Testing complete!!!")
				print(acc)
				print(result_dict)
				self.log(str(result_dict))
				self.log(str(acc))
			elif self.ood:
				self.ood_genset_creation(filenames.external_file)
			else:
				acc = self.validate()
				print("Testing complete!!!")
				print(acc)
				self.log(acc)

		print('Data : ',  data_name)
		self.log(data_name)
		print("Done!")

	def load_examples(self,data_name='Not',save_data=False, n_examples=None):
		'''
		Set n_examples to some positive integer to only load (up to) that 
		number of examples
		'''
		self.log('Loading examples')
		if self.filename is None:
			raise ValueError('Filename argument to constructor can\'t be None')

		self.vocab_to_ints = {}
		self.ints_to_vocab = {}
		self.pos_to_ints={}
		self.ints_to_pos={}
		self.opp_POS={'VBZ':'VBP', 'VBP':'VBZ'}
		# note that 0 in the ints of POS means that the class is reject! that is off NULL TOKEN! 

		examples = []
		n = 0

		deps = deps_from_tsv(self.filename, limit=n_examples)

		for dep in deps:
			tokens = dep['sentence'].split()

			if len(tokens) > self.maxlen or not self.criterion(dep):
				continue

			tokens, POS_tags = self.process_single_dependency(dep, True)

			if dep['label'] == 'ungrammatical':  # this will only operate in the case of Gram. In case of PVN, it wont be there automatticaly 
				POS_tags[int(dep['verb_index']) - 1] = self.opp_POS[POS_tags[int(dep['verb_index']) - 1]]

			ints = []
			sent_POS = []
			for token in tokens:
				if token not in self.vocab_to_ints:
					# zero is for pad
					x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
					self.ints_to_vocab[x] = token
				ints.append(self.vocab_to_ints[token])

			for pos in POS_tags:
				if pos not in self.pos_to_ints:
					x = self.pos_to_ints[pos] =  len(self.pos_to_ints) + 1
					self.ints_to_pos[x] = pos 
				sent_POS.append(self.pos_to_ints[pos])					

			examples.append((self.class_to_code[dep['label']], ints, sent_POS, dep))
			n += 1
			if n_examples is not None and n >= n_examples:
				break

		if (save_data) :
			with open('plus5_v2i.pkl', 'wb') as f:
				pickle.dump(self.vocab_to_ints, f)
			with open('plus5_i2v.pkl', 'wb') as f:
				pickle.dump(self.ints_to_vocab, f)
			with open('plus5_p2i.pkl', 'wb') as f:
				pickle.dump(self.pos_to_ints, f)
			with open('plus5_i2p.pkl', 'wb') as f:
				pickle.dump(self.ints_to_pos, f)

		return examples

	def load_model(self):
		self.model = torch.load(self.model_name)   

	def input_to_string(self, x_input):
		#x_input is the example we want to convert to the string 
		#x_input should be in the form of 1D list. 
		example_string = ""
		for token in x_input:
			if token == 0:
				continue
			str_tok =  self.ints_to_vocab[token]
			example_string+=str_tok+" "
		return example_string

	def topK_acc(self, score):
		# score will be a sorted list, and we will take first K entries. 
		locs, _ =  zip(*score)
		for k in self.K_dict.keys():
			a = list(locs[0:k])
			for loc in a:
				if loc in self.L_dict.keys():
					self.K_dict[k]+=1
					break


	def update_attn_dict(self, attention_weights, verb_loc, x_test):
		# attention_weights (bsz, input_sent_len) Get the list for every exampple in batch Having a tuple 
		# convert all attention weights and verb loc to numpy 
		bsz = x_test.size(0)
		verb_index =  verb_loc -1 
		verb_index = verb_index.squeeze().cpu().numpy()
		attention_weights = attention_weights.cpu().numpy() # (bsz, len)
		x_test =  x_test.cpu()
		input_len =  [len(self.input_to_string(x_test[i].tolist()).split()) for i in range(bsz)]
		per_batch_corr=0   # if top-K contains element from the set {0, 1, ... L-1}, then we increase the count 
		for i in range(bsz):
			# flag=False
			with_indices = list(attention_weights[i, -input_len[i]:])
			for j in range(len(with_indices)):
				with_indices[j] = (j, with_indices[j])
			with_indices.sort(key = lambda x: x[1] , reverse=True)  # sorting in descending order with respect to attention weights 
			with_indices =  [(j[0]-verb_index[i], j[1]) for j in with_indices] # a list containing (location wrt verb and probablity)
			self.topK_acc(with_indices)
			for j in with_indices:
				if j[0] in self.attn_dist.keys():
					avg_prob, tot =  self.attn_dist[j[0]]
					avg_prob =  float(avg_prob*tot + j[1])/float(tot+1)
					tot+=1
					self.attn_dist[j[0]] = (avg_prob, tot)
				else:
					self.attn_dist[j[0]] = (j[1], 1)

	def compare_models(self, m1,  m2):
		# to compare the accuracies of 2 models and log where the 2 are producing different outputs.
		x_diff_pred = []
		ground_truth=[]
		predicted1= []
		predicted2= []
		verb_location=[]
		accuracy_1 = 0
		accuracy_2 = 0
		batches_processed=0
		example_processed=0
		with torch.no_grad():
			m1.eval()
			m2.eval()
			for x_test, y_test, verb_loc_train in self.Test_DataGenerator:
				bsz = x_test.size(0)    			
				x_test = x_test.view(bsz, self.maxlen)
				y_test = y_test.view(bsz, )
				example_processed+=bsz
				output1 =  m1.predict(x_test, y_test)
				output2 =  m2.predict(x_test, y_test)
				pred1 =  output1[1].squeeze()
				pred2 = output2[1].squeeze()
				accuracy_1+=output1[2]
				accuracy_2+=output2[2]
				for j in range(len(pred1)):
					if pred1[j]!=pred2[j]:
						x_diff_pred.append(self.input_to_string(x_test[j].tolist()))
						ground_truth.append(y_test[j].tolist())
						predicted1.append(pred1[j].tolist())
						predicted2.append(pred2[j].tolist())
						verb_location.append(verb_loc_train[j].tolist())

				batches_processed+=1
				if batches_processed%50==0:
					print("batches processed {}/{}".format(batches_processed, len(self.Test_DataGenerator)))
		d= {'sent':x_diff_pred, 'ground_truth':ground_truth, 'verbLoc': verb_location, 'pred1':predicted1, 'pred2':predicted2}	
		with open("different_sent_dump.csv", "w") as outfile:
		   writer = csv.writer(outfile)
		   writer.writerow(d.keys())
		   writer.writerows(zip(*d.values()))
		msg = "Accuracy of Model1 is {} \t Accuracy for Model 2 is {}".format(float(accuracy_1)/float(example_processed), float(accuracy_2)/float(example_processed))
		print(msg)
		self.log(msg)

	def validate(self):
		self.attn_dist={}   # A dict to be used to update the average distribution over the testing examples!
		# self.per_loc_probab_dist={}
		self.L_dict = {i:"" for i in range(self.L)}
		self.K_dict = {i:0 for i in range(1, self.K+1)}
		if not self.train:
			print("Total testing Dataset size {}".format(len(self.Test_DataGenerator.dataset)))
		verbose=True
		correct=0
		self.specific = []	
		hidden_last=[]
		y_out=[]
		example_processed=0
		self.model.eval()
		with torch.no_grad():
			loss_ = 0
			counter=0
			for x_test, y_test, verb_loc_test in self.Test_DataGenerator:
				bsz = x_test.size(0)
				x_test = x_test.view(bsz, self.maxlen)
				y_test = y_test.view(bsz, )
				example_processed+=bsz
				if self.act_attention or self.max_attn:
					loss, _, _, acc, attention_weights, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 
				else:
					loss, _, _, acc, h_n, h_last = self.model.predict(x_test, y_test, compute_loss=True) 					
				loss_+=loss*bsz
				correct+=acc
				hidden_last.append((h_n[-1]))
				y_out.append(y_test)
				if not self.train and (self.act_attention or self.max_attn):
					self.update_attn_dict(attention_weights, verb_loc_test, x_test)
					counter+=1
	

		# dump distributions!! 
		if not self.train and (self.act_attention or self.max_attn) :
			with open('attn_dist.pickle', 'wb') as f:
				pickle.dump(self.attn_dist, f)
			self.K_dict = {k: float(self.K_dict[k])/float(example_processed) for k in self.K_dict.keys()}
			print("Top k results for L = {}".format(self.L))
			print(str(self.K_dict))
			self.log("Top k results for L = {}".format(self.L))
			self.log(str(self.K_dict))
			self.log(str(self.attn_dist))
			print("Following is average attention weight distribution for the testing examples! 0 key means verb loc")
			print(self.attn_dist) 

		hidden_last = torch.cat(hidden_last, dim=0).tolist()
		y_out = torch.cat(y_out, dim=0).tolist()
		with open('hidden_last.pkl', 'wb') as f:
			pickle.dump((hidden_last, y_out), f)

		self.model.train()
		return float(correct)/float(example_processed), float(loss_)/len(self.Test_DataGenerator.dataset)
		

	def detach_states(self, states):
		return torch.stack([states[i].detach() for i in range(states.size(0))]) # per layer detach the hidden states

	# def save_weights(self):
	# 	#saves the recurrent weights
	# 	weight_list=[]
	# 	weight_dict={}
	# 	for layer in range(self.num_layers):
	# 		if self.rnn_arch!= 'LSTM' or self.rnn_arch!='GRU'or self.rnn_arch!='RNN':
	# 			weight_list.append(self.model.recurrent_layer.weight_hh_l[layer].view(self.hidden_size,))
	# 		else:
	# 			weight_list.append(self.model.recurrent_layer.get_cell(layer).weight_hh.view(self.hidden_size,))

	# 	with open('weight_hh.pkl', 'wb') as f:
	# 		pickle.dump(weight_list, f)


	def train_model(self):
		self.demarcate_train=True 
		self.log('Training')
		print('Training')

		if not hasattr(self, 'model'):
			self.create_model()

		self.log("Total size of training set {}".format(len(self.X_train)))
		print("Total size of training set {}".format(len(self.X_train)))

		self.log("Total Training epochs {}".format(self.num_epochs))
		print("Total Training epochs {}".format(self.num_epochs))


		max_acc= 0; min_loss = float("Inf")
		optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
		if self.annealing:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True, eps=1e-6)



		for epoch in range(self.num_epochs) :
			h_n = None
			flag=True
			self.log('epoch : ' + str(epoch))
			self.log_grad('epoch : ' + str(epoch))
			self.log_alpha('epoch : ' + str(epoch))
			index=0
			total_batches = len(self.Train_DataGenerator)
			epoch_loss = 0
			start_time = time.time()
			for x_batch, y_batch, _ in self.Train_DataGenerator :
				bsz = x_batch.size(0)
				x_batch = x_batch.view(bsz, self.maxlen).to(device)
				y_batch = y_batch.view(bsz,).to(device)
				self.model.zero_grad()
				if self.act_attention or self.max_attn:
					if self.use_hidden:
						if flag:
							loss, _, _, _, attention_weights,  h_n, h_last = self.model.predict(x_batch, y_batch, h_n, compute_loss=True)
							flag = False
						else:
							loss, _, _, _, attention_weights,  h_n, h_last = self.model.predict(x_batch, y_batch, h_n[:, 0:bsz, :], compute_loss=True)						
					else:
						loss, _, _, _, _,  _, _ = self.model.predict(x_batch, y_batch, compute_loss=True)
				else:	
					if self.use_hidden:
						if flag:
							loss, _, _, _, h_n, h_last = self.model.predict(x_batch, y_batch, h_n, compute_loss=True)
							flag = False
						else:
							loss, _, _, _, h_n, h_last = self.model.predict(x_batch, y_batch, h_n[:, 0:bsz, :], compute_loss=True)						
					else:
						loss, _, _, _, _, _ = self.model.predict(x_batch, y_batch, compute_loss=True)

				epoch_loss+=loss.item()
				loss.backward()	
				if self.use_hidden:
					h_n = self.detach_states(h_n)
				optimizer.step()
				index+=1

				if (index)%30 == 0:
					elapsed_time=time.time()-start_time
					acc, val_loss = self.validate()
					if (acc >= max_acc) :
						model_name = self.model_prefix + '.pkl'
						torch.save(self.model, model_name)
						max_acc = acc
					if val_loss < min_loss:
						min_loss = val_loss
					acc, result_dict = self.result_demarcated()
					self.log_demarcate_train(acc)
					# self.log_demarcate_train(result_dict)
					self.demarcate_train = not self.demarcate_train
					acc, result_dict = self.result_demarcated()
					self.log_demarcate_val(acc)
					# self.log_demarcate_val(result_dict)
					self.demarcate_train = not self.demarcate_train
					counter = 0
					args=[index, total_batches, epoch, self.num_epochs, max_acc, epoch_loss/index, min_loss, float(index)/float(elapsed_time)]
					self.log("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))
					self.log_grad("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))
					print("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))
					for param in self.model.parameters():
						if param.grad is not None:
							self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
							counter += 1
				for name,param in self.model.named_parameters(): 
					for i in range(self.num_layers):               
						if name=="recurrent_layer.cell_{}.rgate".format(i):
							self.log_alpha(str((param.data)))


			acc, val_loss = self.validate()   # this is fraction based accuracy 
			if self.annealing:
				scheduler.step(val_loss)

			if val_loss < min_loss:
				min_loss = val_loss

				args=[index, total_batches, epoch, self.num_epochs, max_acc, epoch_loss/index, min_loss, float(index)/float(elapsed_time)]
				self.log("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))
				self.log_grad("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))
				print("Total bsz done {}/{} || Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f} || Speed {} b/s".format(*args))

			if (acc > max_acc) :
				model_name = self.model_prefix + '.pkl'
				torch.save(self.model, model_name)
				max_acc = acc

			acc, result_dict = self.result_demarcated()
			self.log_demarcate_train(acc)
			# self.log_demarcate_train(result_dict)
			self.demarcate_train = not self.demarcate_train
			acc, result_dict = self.result_demarcated()
			self.log_demarcate_val(acc)
			# self.log_demarcate_val(result_dict)
			self.demarcate_train = not self.demarcate_train

			index=0; epoch_loss=0


		print("End of training !!!")


	def create_model_POS(self):
		self.log('Creating model POS')
		self.log('vocab size : ' + str(len(self.vocab_to_ints)))
		print('Creating model POS')
		print('vocab size : ' + str(len(self.vocab_to_ints)))
		self.model = POS_Tagger(rnn_arch= self.rnn_arch, embedding_size = self.embedding_size, hidden_size=self.hidden_size, vocab_size = len(self.vocab_to_ints)+1, num_layers=self.num_layers, output_size=len(self.pos_to_ints)+1, dropout=self.dropout, act_attention=self.act_attention, max_attn=self.max_attn, num_heads=self.nheads, activation=self.activation)
		self.model = self.model.to(device)

	def validate_tagger(self):
		correct=0
		tokens_processed=0
		self.model.eval()
		with torch.no_grad():
			loss_ = 0
			counter=0
			for x_test, y_test, verb_loc_test in self.Test_DataGenerator:
				bsz = x_test.size(0)
				x_test = x_test.view(bsz, self.maxlen)
				y_test = y_test.view(bsz, self.maxlen)
				tokens_processed+=bsz*self.maxlen
				loss, _, _, acc, _, _ = self.model.predict(x_test, y_test, compute_loss=True) 					
				loss_+=loss*bsz*self.maxlen
				correct+=acc

		final_accuracy = float(correct)/float(tokens_processed)
		loss =  float(loss_)/float(tokens_processed)

		return final_accuracy, loss 

	def train_tagging(self):
		self.log('Training POS tagger!')
		print('Training')

		if not hasattr(self, 'model'):
			self.create_model_POS()

		self.log("Total size of training set {}".format(len(self.X_train)))
		print("Total size of training set {}".format(len(self.X_train)))

		self.log("Total Training epochs {}".format(self.num_epochs))
		print("Total Training epochs {}".format(self.num_epochs))


		max_acc= 0; min_loss = float("Inf")
		optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
		if self.annealing:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True, eps=1e-6)

		for epoch in range(self.num_epochs) :
			h_n = None
			flag=True
			self.log('epoch : ' + str(epoch))
			self.log_grad('epoch : ' + str(epoch))
			self.log_alpha('epoch : ' + str(epoch))
			index=0
			total_batches = len(self.Train_DataGenerator)
			epoch_loss = 0

			for x_batch, y_batch, _ in self.Train_DataGenerator :
				bsz = x_batch.size(0)
				x_batch = x_batch.view(bsz, self.maxlen).to(device)
				y_batch = y_batch.view(bsz, self.maxlen).to(device)
				
				self.model.zero_grad()
				if self.use_hidden:
					if flag:
						loss, _, _, _, h_n, h_last = self.model.predict(x_batch, y_batch, h_n, compute_loss=True)
						flag = False
					else:
						loss, _, _, _, h_n, h_last = self.model.predict(x_batch, y_batch, h_n[:, 0:bsz, :], compute_loss=True)						
				else:
					loss, _, _, _, _, _ = self.model.predict(x_batch, y_batch, compute_loss=True)

				epoch_loss+=loss.item()*bsz 
				loss.backward()	
				if self.use_hidden:
					h_n = self.detach_states(h_n)
				optimizer.step()
				index+=1

				if (index)%30 == 0:
					acc, val_loss = self.validate()
					if (acc >= max_acc) :
						model_name = self.model_prefix + '.pkl'
						torch.save(self.model, model_name)
						max_acc = acc
					if val_loss < min_loss:
						min_loss = val_loss

					counter = 0
					self.log("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/(index*bsz), min_loss))
					self.log_grad("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/(index*bsz), min_loss))
					print("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/(index*bsz), min_loss))
					for param in self.model.parameters():
						if param.grad is not None:
							self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
							counter += 1
				for name,param in self.model.named_parameters(): 
					for i in range(self.num_layers):               
						if name=="recurrent_layer.cell_{}.rgate".format(i):
							self.log_alpha(str((param.data)))


			acc, val_loss = self.validate()   # this is fraction based accuracy 
			if self.annealing:
				scheduler.step(val_loss)

			if val_loss < min_loss:
				min_loss = val_loss

			print("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/len(self.Train_DataGenerator.dataset), min_loss))
			self.log("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/len(self.Train_DataGenerator.dataset), min_loss))
			self.log_grad("Total Epochs Done {}/{} || Max Validation Accuracy {:.4f} || Epoch loss {:.4f} || Min loss {:.4f}".format(epoch, self.num_epochs, max_acc, epoch_loss/len(self.Train_DataGenerator.dataset), min_loss))

			if (acc > max_acc) :
				model_name = self.model_prefix + '.pkl'
				torch.save(self.model, model_name)
				max_acc = acc

			index=0; epoch_loss=0


		print("End of training !!!")

