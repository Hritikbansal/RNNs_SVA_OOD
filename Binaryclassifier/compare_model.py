from agreement_acceptor_decay_rnn import FullGramSentence, PredictVerbNumber
import filenames
train=False
fullGram=True 

if fullGram:
	pvn = FullGramSentence("ABDRNN", filenames.deps, embedding_size=50, hidden_size=50, output_size=2, prop_train=0.1, output_filename='output_log.txt')

	if train:
		pvn.pipeline(True,train_bsz=128, test_bsz= 2048, model="decay_fullGram.pkl", load_data=True,load=False,num_epochs=10, model_prefix='decay_fullGram', 
					data_name='fullGram', test_size=7000, lr=0.01, annealing=True)
	else:	
		# for l in [1, 2, 3, 4, 5]:
		# 	print(l)
			pvn.pipeline(False,train_bsz=128, test_bsz= 2048, model="decay_fullGram.pkl", load_data=True,load=True,num_epochs=10, model_prefix='decay_fullGram', 
						data_name='fullGram', test_size=0, lr=0.01, annealing=True, compare_models=True, m1='abdrnn_fg.pkl', m2='abdrnn_fg_attn.pkl')

else:
	pvn = PredictVerbNumber("ABDRNN", filenames.deps, embedding_size=50, hidden_size=50, output_size=2, prop_train=0.1, output_filename='output_log.txt')

	if train:
		pvn.pipeline(True,train_bsz=128, test_bsz= 2048, model="decay_pvn.pkl", load_data=False,load=False,num_epochs=10, model_prefix='decay_pvn', 
					data_name='fullGram', test_size=7000, lr=0.01, annealing=True)
	else:	
		# for l in [1, 2, 3, 4, 5]:
		# 	print(l)
			pvn.pipeline(False,train_bsz=128, test_bsz= 2048, model="decay_pvn.pkl", load_data=False,load=True,num_epochs=10, model_prefix='decay_pvn', 
						data_name='fullGram', test_size=0, lr=0.01, annealing=True)
