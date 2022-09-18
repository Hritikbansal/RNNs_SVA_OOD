from agreement_acceptor_decay_rnn import FullGramSentence, PredictVerbNumber
import filenames
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LSTM',
					help='Choice of Model.')
parser.add_argument('--train', action='store_true', default=False,
					help='training/testing')
parser.add_argument('--train_file', type=str, default='train_nat.tsv',
					help='specify training file, testing file gets preprocessed in the natural setting by default')
parser.add_argument('--save_data', action='store_true', default=False,
					help='whether to save preprocessed train and test data')
parser.add_argument('--demarcate_train', action='store_true', default=False,
					help='training/testing')
parser.add_argument('--test_demarcated', action='store_true', default=False,
					help='if want demarcated results')
parser.add_argument('--prop_train', type=float, default=0.896,
					help='amount of training data.')
parser.add_argument('--validation_size', type=int, default=1,
					help='amount of validation data.')
parser.add_argument('--hidden_size', type=int, default=50,
					help='hidden units')
parser.add_argument('--embedding_size', type=int, default=50,
					help='embedding size ')
parser.add_argument('--num_layers', type=int, default=1,
					help='num_layers')
parser.add_argument('--bsz', type=int, default=64,
					help='train_bsz')
parser.add_argument('--epochs', type=int, default=20,
					help='amount of validation data.')
parser.add_argument('--fullGram', action='store_true', default=True,
					help='training/testing')
parser.add_argument('--augment_train', action='store_true', default=False,
					help='augment training set !!!')
parser.add_argument('--augment_test', action='store_true', default=False,
					help='augment testing set!!! ')
parser.add_argument('--domain_adaption', action='store_true', default=False,
					help='use selective setting for training!!! ')
parser.add_argument('--intermediate', action='store_true', default=False,
					help='use intermediate setting for training!!! ')
parser.add_argument('--domain_adaption2', action='store_true', default=False,
					help='use selective setting based on no. if intervening nouns for training!!! ')
parser.add_argument('--ood', action='store_true', default=False,
					help='OOD Dataset-- Linzen 18 !!! ')
parser.add_argument('--act_attention', action='store_true', default=False,
					help='use contrastive train set for testing!!! ')
parser.add_argument('--verb_embedding', action='store_true', default=False,
					help='use contrastive train set for testing!!! ')
# Range of prop_train: 0.7, 0.8, 0.9, 0.99
args = parser.parse_args()


train=args.train
fullGram=args.fullGram

if fullGram:
	pvn = FullGramSentence(args.model, filenames.deps, embedding_size=args.embedding_size, hidden_size=args.hidden_size, output_size=2, 
                        num_layers=args.num_layers, prop_train=args.prop_train, output_filename='output_log.txt')

	if train:
		pvn.pipeline(True, train_bsz=args.bsz, test_bsz= 4096, model=args.model + "_fullGram.pkl", load_data=True,load=False,num_epochs=args.epochs, 
               model_prefix=args.model + "_fullGram", data_name='fullGram', test_size=args.validation_size, lr=0.005 if args.model=="LSTM" or "ONLSTM" else 0.01, annealing=True,
               act_attention=args.act_attention,domain_adaption=args.domain_adaption, intermediate=args.intermediate, domain_adaption2=args.domain_adaption2, filen=args.train_file, ood=args.ood, augment_train=args.augment_train, 
               augment_test=args.augment_test, save_data=args.save_data)
	else:	
		for l in [1]:#, 2, 3, 4, 5]:
			print(l)
			pvn.pipeline(False,train_bsz=128, test_bsz= 2048, model=args.model + "_fullGram.pkl", load_data=True,load=True,num_epochs=10, 
                model_prefix=args.model + "_fullGram", data_name='fullGram', test_size=args.validation_size, lr=0.005 if args.model=="LSTM" or "ONLSTM" else 0.01, annealing=True,act_attention=False,
                domain_adaption=args.domain_adaption,test_demarcated=args.test_demarcated,demarcate_train=args.demarcate_train, L=l, K=5, 
                augment_train=args.augment_train, augment_test=args.augment_test, verb_embedding=args.verb_embedding)

else:
	pvn = PredictVerbNumber(args.model, filenames.deps, embedding_size=50, hidden_size=50, output_size=2, prop_train=0.1, 
                         output_filename='output_log.txt')

	if train:
		pvn.pipeline(True,train_bsz=128, test_bsz= 2048, model="lstm_pvn.pkl", load_data=False,load=False,num_epochs=10, model_prefix='lstm_pvn', 
					data_name='fullGram', test_size=7000, lr=0.01, annealing=True,act_attention=False, save_data=True)
	else:	
		for l in [1, 2, 3, 4, 5]:
			print(l)
			pvn.pipeline(False,train_bsz=128, test_bsz= 2048, model="lstm_pvn.pkl", load_data=False,load=True,num_epochs=10, 
                model_prefix='lstm_pvn', data_name='fullGram', test_size=0, lr=0.01, annealing=True,act_attention=False, L=l, K=5)
