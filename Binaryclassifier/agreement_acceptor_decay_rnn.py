import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import sys
from decay_rnn_model import DECAY_RNN_Model
import six
import pickle
from utils import gen_inflect_from_vocab, dependency_fields, dump_dict_to_csv
from fullgram_model import Agreement_model

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

class RNNAcceptor(DECAY_RNN_Model):

	def verb_emb(self):
		embeddings = []
		count=0; limit=20000
		with torch.no_grad():
			self.model.eval()
			for dep in self.deps_test:
				tokens = dep['sentence'].split()
				v = dep['verb_index']-1
				verb = tokens[v]
				verb_inf = self.inflect_verb[verb]
				verb_int = torch.tensor(self.vocab_to_ints[verb], dtype=torch.long).to(device)
				verb_inf_int = torch.tensor(self.vocab_to_ints[verb_inf], dtype=torch.long).to(device)
				emb_v = self.model.embedding_layer(verb_int)
				emb_inf_v = self.model.embedding_layer(verb_inf_int)
				embeddings.append((emb_v, emb_inf_v))
				count+=1
				if count==limit:
					break

			with open('emb.pkl', 'wb') as f:
				pickle.dump(embeddings, f)

		sys.exit()


	def create_inflected_from_dep(self, X, Y, POS, deps):
		# for time being, we are not considering POS tags!!! Please do not use them...
		deps_flip = copy.deepcopy(deps)
		X_flip = copy.deepcopy(X)
		Y_flip = copy.deepcopy(Y)
		POS_flip =  copy.deepcopy(POS)
		self.opp_POS={'VBZ':'VBP', 'VBP':'VBZ'}
		self.opposite_ = {'grammatical':'ungrammatical', 'ungrammatical':'grammatical'}
		for i in range(len(X_flip)):
			verb_idx = self.maxlen - len(deps_flip[i]['sentence'].split()) + int(deps_flip[i]['verb_index']) - 1
			verb = self.ints_to_vocab[X_flip[i][verb_idx]]
			flip_v = self.inflect_verb[verb]
			X_flip[i][verb_idx]  = self.vocab_to_ints[flip_v] # Now we have changed the verb
			Y_flip[i] = int(not bool(Y_flip[i]))	
			deps_flip[i]['label'] = self.opposite_[deps_flip[i]['label']]	

		return X_flip, Y_flip, POS_flip, deps_flip

	def create_inflected_train_test(self):
		X_train, Y_train, POS_train, deps_train =  self.create_inflected_from_dep(self.X_train, self.Y_train, self.POS_train, self.deps_train)
		X_test,  Y_test, POS_test, deps_test =  self.create_inflected_from_dep(self.X_test, self.Y_test, self.POS_test, self.deps_test)
		mixup_size = int(self.augment_ratio*len(X_train))

		if self.augment_train:
			print("Augmenting {} train examples!!".format(mixup_size))
			self.X_train = list(self.X_train)+list(X_train)[:mixup_size] # augmenting the flipped dataset!!! 
			self.Y_train = np.concatenate((self.Y_train, Y_train[:mixup_size]), axis=0)
			self.deps_train =   list(self.deps_train)+list(deps_train)[:mixup_size]
			self.POS_train =  list(self.POS_train)+list(POS_train)[:mixup_size]

		mixup_size = int(self.augment_ratio*len(X_test))

		if self.augment_test:
			print("Augmenting {} test examples!!".format(mixup_size))
			self.X_test = list(self.X_test)+list(X_test)[:mixup_size] # augmenting the flipped dataset!!! 
			self.Y_test = np.concatenate((self.Y_test, Y_test[:mixup_size]), axis=0)
			self.deps_test =   list(self.deps_test)+list(deps_test)[:mixup_size]
			self.POS_test =  list(self.POS_test)+list(POS_test)[:mixup_size]			

		print("Size of X_{} is {}".format('train', len(self.X_train)))
		print("Size of X_{} is {}".format('test', len(self.X_test)))
		print("Size of Y_{} is {}".format('train', len(self.Y_train)))
		print("Size of Y_{} is {}".format('test', len(self.Y_test)))

		print('Data preparation done!')

	def create_train_and_test(self, examples, test_size, data_name, save_data=False):
		d = [[], []]
		for i, s, pos, dep in examples:
			d[i].append((i, s, pos, dep))
		random.seed(1)
		random.shuffle(d[0])
		random.shuffle(d[1])
		if self.equalize_classes:
			l = min(len(d[0]), len(d[1]))
			examples = d[0][:l] + d[1][:l]
		else:
			examples = d[0] + d[1]
		random.shuffle(examples)

		Y, X, POS, deps = zip(*examples)
		Y = np.asarray(Y)
		X = pad_sequences(X, maxlen = self.maxlen)
		POS = pad_sequences(POS, maxlen=self.maxlen)


		n_train = int(self.prop_train * len(X))
		# self.log('ntrain', n_train, self.prop_train, len(X), self.prop_train * len(X))
		self.X_train, self.Y_train = X[:n_train], Y[:n_train]
		self.POS_train = POS[:n_train]
		self.deps_train = deps[:n_train]
		if (test_size > 0) :
			self.X_test, self.Y_test = X[n_train : n_train+test_size], Y[n_train : n_train+test_size]
			self.deps_test = deps[n_train : n_train+test_size]
			self.POS_test = POS[n_train : n_train+test_size]
		else :
			self.X_test, self.Y_test = X[n_train:], Y[n_train:]
			self.deps_test = deps[n_train:]
			self.POS_test = POS[n_train:]

		self.create_inflected_train_test()

		if (save_data) :
			with open('X_' + data_name + '_data.pkl', 'wb') as f:
				pickle.dump(X, f)
			with open('Y_' + data_name + '_data.pkl', 'wb') as f:
				pickle.dump(Y, f)
			with open('deps_' + data_name + '_data.pkl', 'wb') as f:
				pickle.dump(deps, f)
			with open('POS_' + data_name + '_data.pkl', 'wb') as f:
				pickle.dump(POS, f)

			print("Exiting, saved the data!!!!")
			sys.exit()


	def load_train_and_test(self, test_size, data_name, domain_adaption=False):
		# Y = np.asarray(Y)
		# X = pad_sequences(X, maxlen = self.maxlen)

		with open('../grammar_data/' + data_name + '_v2i.pkl', 'rb') as f:
			self.vocab_to_ints = pickle.load(f)

		with open('../grammar_data/' + data_name + '_i2v.pkl', 'rb') as f:
			self.ints_to_vocab = pickle.load(f)

		with open('../grammar_data/' + data_name + '_p2i.pkl', 'rb') as f:
			self.pos_to_ints = pickle.load(f)

		with open('../grammar_data/' + data_name + '_i2p.pkl', 'rb') as f:
			self.ints_to_pos = pickle.load(f)

		X = []
		Y = []

		with open('../grammar_data/X_' + data_name + '_data.pkl', 'rb') as f:
			X = pickle.load(f)

		with open('../grammar_data/Y_' + data_name + '_data.pkl', 'rb') as f:
			Y = pickle.load(f)

		with open('../grammar_data/deps_' + data_name + '_data.pkl', 'rb') as f:
			deps = pickle.load(f)

		with open('../grammar_data/POS_' + data_name + '_data.pkl', 'rb') as f:
			POS = pickle.load(f)

		n_train = int(self.prop_train * len(X))
		if(domain_adaption):
			ind = []
			for i in range(n_train):
				if(deps[i]['n_diff_intervening']>1):
					ind.append(i)
			ind = np.array(ind)
			print('length of training set: '+str(len(ind)))
			self.X_train, self.Y_train = X[ind], Y[ind]
			self.POS_train = POS[ind]
			self.deps_train = np.array(deps)[ind].tolist()
		else:
			self.X_train, self.Y_train = X[:n_train], Y[:n_train]
			self.POS_train = POS[:n_train]
			self.deps_train = deps[:n_train]

		if (test_size > 0) :
			if(domain_adaption):
				ind = []
				for i in range(n_train,len(deps)):
					if(deps[i]['n_diff_intervening']>0):
				 		ind.append(i)
				ind = np.array(ind)
				print('length of validation set: '+str(len(ind)))
				self.X_test, self.Y_test = X[ind], Y[ind]
				self.deps_test = np.array(deps)[ind].tolist()
				self.POS_test = POS[ind]
			else:
				self.X_test, self.Y_test = X[n_train : n_train+test_size], Y[n_train : n_train+test_size]
				self.deps_test = deps[n_train : n_train+test_size]
				self.POS_test = POS[n_train : n_train+test_size]
		else :
			if(domain_adaption):
				ind = []
				for i in range(int(0.95*len(deps)), len(deps)):
					if(deps[i]['n_diff_intervening']<1):
						ind.append(i)
				ind = np.array(ind)
				print('length of test set: '+str(len(ind)))
				self.X_test, self.Y_test = X[ind], Y[ind]
				self.deps_test = np.array(deps)[ind].tolist()
				self.POS_test = POS[ind]
			else:
				self.X_test, self.Y_test = X[n_train:], Y[n_train:]
				self.deps_test = deps[n_train:]
				self.POS_test = POS[n_train:]

		self.create_inflected_train_test()

class PredictVerbNumber(RNNAcceptor):

	def __init__(self, *args, **kwargs):
		RNNAcceptor.__init__(self, *args, **kwargs)
		self.class_to_code = {'VBZ': 0, 'VBP': 1}
		self.code_to_class = {x: y for y, x in self.class_to_code.items()}

	def process_single_dependency(self, dep):
		dep['label'] = dep['verb_pos']
		v = int(dep['verb_index']) - 1
		tokens = dep['sentence'].split()[:v]
		return tokens

class InflectVerb(PredictVerbNumber):
	'''
	Present all words up to _and including_ the verb, but withhold the number
	of the verb (always present it in the singular form). Supervision is
	still the original number of the verb. This task allows the system to use
	the semantics of the verb to establish the dependency with its subject, so
	may be easier. Conversely, this may mess up the embedding of the singular
	form of the verb; one solution could be to expand the vocabulary with
	number-neutral lemma forms.
	'''

	def __init__(self, *args, **kwargs):
		super(InflectVerb, self).__init__(*args, **kwargs)
		self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

	def process_single_dependency(self, dep):
		dep['label'] = dep['verb_pos']
		v = int(dep['verb_index']) - 1
		tokens = dep['sentence'].split()[:v+1]
		if dep['verb_pos'] == 'VBP':
			tokens[v] = self.inflect_verb[tokens[v]]
		return tokens

class CorruptAgreement(RNNAcceptor):

	def __init__(self, *args, **kwargs):
		RNNAcceptor.__init__(self, *args, **kwargs)
		self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
		self.code_to_class = {x: y for y, x in self.class_to_code.items()}
		self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

	def process_single_dependency(self, dep):
		tokens = dep['sentence'].split()
		if random.random() < 0.5:
			dep['label'] = 'ungrammatical'
			v = int(dep['verb_index']) - 1
			tokens[v] = self.inflect_verb[tokens[v]]
			dep['sentence'] = ' '.join(tokens)
		else:
			dep['label'] = 'grammatical'
		return tokens


class GrammaticalHalfSentence(RNNAcceptor):

	def __init__(self, *args, **kwargs):
		RNNAcceptor.__init__(self, *args, **kwargs)
		self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
		self.code_to_class = {x: y for y, x in self.class_to_code.items()}
		self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

	def process_single_dependency(self, dep):
		tokens = dep['sentence'].split()
		v = int(dep['verb_index']) - 1
		tokens = tokens[:v+1]
		if random.random() < 0.5:
			dep['label'] = 'ungrammatical'
			tokens[v] = self.inflect_verb[tokens[v]]
		else:
			dep['label'] = 'grammatical'
		dep['sentence'] = ' '.join(tokens[:v+1])
		return tokens

class GramHalfPlusSentence(RNNAcceptor):

	def __init__(self, *args, **kwargs):
		RNNAcceptor.__init__(self, *args, **kwargs)
		self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
		self.code_to_class = {x: y for y, x in self.class_to_code.items()}
		self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

	def process_single_dependency(self, dep):
		tokens = dep['sentence'].split()
		v = int(dep['verb_index']) - 1
		tokens = tokens[:v+1 + self.len_after_verb]
		if random.random() < 0.5:
			dep['label'] = 'ungrammatical'
			tokens[v] = self.inflect_verb[tokens[v]]
		else:
			dep['label'] = 'grammatical'
		dep['sentence'] = ' '.join(tokens[:v+1 + self.len_after_verb])
		return tokens

class FullGramSentence(RNNAcceptor):

	def __init__(self, *args, **kwargs):
		RNNAcceptor.__init__(self, *args, **kwargs)
		self.class_to_code = {'grammatical': 0, 'ungrammatical': 1}
		self.inflect_POS={'VBZ':'VBP', 'VBP':'VBZ'}
		self.code_to_class = {x: y for y, x in self.class_to_code.items()}
		self.inflect_verb, _ = gen_inflect_from_vocab(self.vocab_file)

	def process_single_dependency(self, dep, pos=False):
		tokens = dep['sentence'].split()
		POS =  dep['pos_sentence'].split()
		v = int(dep['verb_index']) - 1
		if random.random() < 0.5:
			dep['label'] = 'ungrammatical'
			tokens[v] = self.inflect_verb[tokens[v]]
			POS[v]=  self.inflect_POS[POS[v]]
		else:
			dep['label'] = 'grammatical'
		if pos:
			return tokens, POS
		else:
			return tokens
