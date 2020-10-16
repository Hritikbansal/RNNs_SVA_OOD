from agreement_acceptor_decay_rnn import FullGramSentence, PredictVerbNumber
import filenames

train=True
pvn = PredictVerbNumber("SDRNN", filenames.deps, embedding_size=50, hidden_size=50, output_size=2, prop_train=0.1, output_filename='output_log.txt')

if train:
	pvn.pipeline(True,train_bsz=128, test_bsz= 2048, model="decay_pvn.pkl", load_data=False,load=False,num_epochs=10, model_prefix='decay_pvn', 
				data_name='fullGram', test_size=7000, lr=0.01, annealing=True,act_attention=False, train_tagger=True)
else:	
	for l in [1, 2, 3, 4, 5]:
		print(l)
		pvn.pipeline(False,train_bsz=128, test_bsz= 2048, model="decay_pvn.pkl", load_data=False,load=True,num_epochs=10, model_prefix='decay_pvn', 
					data_name='fullGram', test_size=0, lr=0.01, annealing=True,act_attention=False, L=l, K=5)
