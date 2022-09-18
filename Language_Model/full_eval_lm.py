import torch
import layers
import argparse
import pickle as pkl
from text.utils import gen_inflect_from_vocab, deps_from_tsv, deps_to_tsv
from itertools import zip_longest
from collections import Counter
import torch.nn.functional as F
from text.dataset import Dataset
import torch.nn as nn
import math
import random

#torch.backends.cudnn.enabled = False



def log(message):
    with open('logs/' + 'output.txt', 'a') as file:
        file.write(str(message) + '\n')

def prepare_batch(mb):
    mb = mb.to(device)
    x = mb[:-1, :].clone()
    y = mb[1:, :].clone()
    return x, y.view(-1)

def build_crit(n_words):
    weight = torch.ones(n_words)
    weight[0] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    return crit.to(device)


def eval(model, valid, crit):
    model.eval()
    valid_nlls = []
    n_words = 0
    with torch.no_grad():
        for i in range(len(valid)):
            x, y = prepare_batch(valid[i])
            log_prob, _ = model(x)
            nll = crit(log_prob, y)
            
            valid_nlls.append(nll.item())
            
      
            n_words += y.ne(0).int().sum().item()
           
     
        model.train()
        nll = torch.FloatTensor(valid_nlls).sum().item() / n_words
    return math.exp(nll)

def eval_model(model, valid):
    crit = build_crit(opt.n_words)
    test_perplexity = eval(model, valid, crit)
    print(test_perplexity)
    log(test_perplexity)


def padding(input_, max_len):
    padding_mask = [(max_len - len(x),0) for x in input_]
    for i in range(len(input_)):
        input_[i] =  F.pad(torch.LongTensor(input_[i]), padding_mask[i])

    return torch.stack(input_).t()

parser = argparse.ArgumentParser(description='Evaluate LM')
parser.add_argument('-checkpoint', required=True,
                    help='file saved trained parameters')
parser.add_argument('-input', required=True, help='input file.')
parser.add_argument('-output', required=True, help='output for plotting.')
parser.add_argument('-batch_size', type=int, default=256,
                    help='batch size, set larger for speed.')
parser.add_argument('--N', type=int, default=15000,
                    help='N examples to save')
parser.add_argument('--rnn_arch', type=str, default='LSTM', 
                    help="Type of RNN architcture to be chosen from LSTM, GRU, ONLSTM, DRNN")
parser.add_argument('--get_hidden', action='store_true',
                    help='Gets hidden representations for comparison!')

opt = parser.parse_args()

checkpoint = torch.load(opt.checkpoint)
saved_opt = checkpoint['opt']
saved_opt.input = opt.input
saved_opt.output = opt.output
saved_opt.rnn_arch =  opt.rnn_arch
saved_opt.batch_size = opt.batch_size
saved_opt.N =  opt.N
saved_opt.get_hidden=opt.get_hidden
opt = saved_opt
print('| reconstruct network')
log('| reconstruct network')

if opt.arch == 'rnn':
    model = layers.RNNLM(opt.word_vec_size, opt.hidden_size, opt.n_words, opt.layers,
                        opt.dropout, rnn_arch=opt.rnn_arch, activation = opt.activation, tied=opt.tied)
elif opt.arch =='RIM':
    print("Building RIM architecture!!!!")
    model = layers.RIM_model(opt.word_vec_size, opt.hidden_size, opt.n_words, opt.layers, opt.num_units, opt.rnn_arch, opt.k,
                         opt.dropout, tied=opt.tied)
else:
    model = layers.Transformer(opt.word_vec_size, opt.n_words,
                               opt.num_heads, opt.head_size,
                               opt.layers, opt.inner_size,
                               opt.dropout, opt.tied)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('| load parameters')
log('| load parameters')
model.load_state_dict(checkpoint['params'])
model.eval()


# build inverse dictionary
w2idx = pkl.load(open(opt.dict, 'rb'))
print('| vocab size: %d' % len(w2idx))
log('| vocab size: %d' % len(w2idx))

inflect_verb, _ = gen_inflect_from_vocab('data/wiki.vocab')

# read data here
deps = deps_from_tsv(opt.input)
dis_hit = Counter()
sing_hit = 0
sing_tot = 0
plur_hit  = 0
plur_tot = 0
dis_tot = Counter()
dif_hit = Counter()
dif_tot = Counter()

bidx = 0
                     
all_hidden=[]
n_list_global=[]
n_diff_list_global=[]
incorrect_deps = []
count_notinvocab = 0
batch_call = 0

with torch.no_grad():
    print("No. of sentences in Testing Set: ",len(deps))
    print("batch size: ",opt.batch_size)
    for i in range(0, len(deps), opt.batch_size):
            mb = []
            bidx += 1
            if bidx % 100 == 0:
                print('process {:5d} / {:d}'.format(i, len(deps)))
                log('process {:5d} / {:d}'.format(i, len(deps)))
                           
            for dep in deps[i: i+opt.batch_size]:
                n = int(dep['n_intervening'])
                n_diff = int(dep['n_diff_intervening'])
                d = int(dep['distance'])
                inflection  =  0 if dep['verb_pos'] == 'VBZ' else 1 
                t_verb = dep['verb']
                #f_verb = inflect_verb[t_verb]
                t_verb_i = w2idx.get(t_verb, 1)
                if t_verb_i == 1:
                    count_notinvocab+=1
                    #print("t_verb ",t_verb, len(t_verb),sep=" ")
                    continue
                f_verb = inflect_verb[t_verb]
                f_verb_i = w2idx.get(f_verb, 1)
                tokens = dep['sentence'].split()
                v = int(dep['verb_index']) - 1
                ws = ['<bos>'] + tokens[:v]
                ws = [w2idx.get(w, 1) for w in ws]
                mb += [(ws, t_verb_i, f_verb_i, d, n, n_diff, inflection, dep)]
            mb.sort(key=lambda t: len(t[0]), reverse=True)
            xs = [x[0] for x in mb]
            tv = [x[1] for x in mb] # list of sentence-verb indices in batch of sentences
            fv = [x[2] for x in mb] # list of sentence-inflected-verb indices in batch of sentences
            n_list = [x[4] for x in mb]
            n_diff_list = [x[5] for x in mb]
            inflection_list = [x[6] for x in mb]
            n_list_global+=n_list
            n_diff_list_global += n_diff_list
        
            xs = padding(xs, len(xs[0])).to(device) # Left padding! 
            if opt.get_hidden:
                batch_call+=1
                scores, hidden = model(xs, True)
                #print("forward calls: ",batch_call)
                #print("scores ",scores)
                if isinstance(hidden, tuple) :
                    hidden = hidden[0]
                all_hidden.append(hidden)
            else:
                scores, _ = model(xs, True)
        
            b = [i for i in range(len(tv))]
            true_scores = scores[b, tv]  # advance indexing
            fake_scores = scores[b, fv]
            corrects = true_scores.gt(fake_scores).view(-1).tolist()
            #print("len of corrects: ",len(corrects))
            for i, v in enumerate(corrects):
                #print("TESTING, corrects: ",corrects,sep=" ")
                dx = mb[i][3]
                inflection =  mb[i][6]
                dis_tot[dx] += 1
                dis_hit[dx] += v
                if v==0:
                    incorrect_deps.append(mb[i][-1])
                if inflection==0:
                    sing_tot+=1
                    sing_hit+=v 
                else:
                    plur_tot+=1
                    plur_hit+=v 
                    
                n0 = mb[i][4] 
                n1 = mb[i][5]
                
                if n0>3: #for accuracy over sentences with more than 3 interveners
                    n0=">3"
                if n1>3: #for accuracy over sentences with more than 3 attractors
                    n1=">3"
        
                #n = (n0, n1) #uncomment this and comment below for accuracy over interveners x attractors
                
                n = n1 
        
                dif_tot[n] += 1
                dif_hit[n] += v

deps_to_tsv(incorrect_deps, 'incorr.tsv')

print("Verbs not in Vocab: ", count_notinvocab)
print("Final number of sentences in test set: ", len(deps) - count_notinvocab)

print(opt.get_hidden)
if opt.get_hidden:
    print('Saving first {} of all last layer hidden(s)'.format(opt.N))
    all_hidden=torch.cat(all_hidden, dim=1) #along batches 
    print(all_hidden.shape)
    shuffle_list = [i for i in range(all_hidden.shape[1])]
    random.shuffle(shuffle_list)
    # assert all_hidden.shape[1]==len(deps)
    with open('all_hidden_1.pkl', 'wb') as f:
        pkl.dump(all_hidden[:, shuffle_list[:opt.N], :], f)
    with open('all_hidden_2.pkl', 'wb') as f:
        pkl.dump(all_hidden[:, shuffle_list[opt.N:2*opt.N], :], f)

dis_acc = {}
dif_acc = {}
print(dis_tot)
log('Accuracy by distance')
print(dis_tot)
log('Accuracy by distance')
print('Accuracy by distance')
for k in sorted(dis_hit.keys()):
    v = dis_hit[k]
    acc = v / dis_tot[k]
    dis_acc[k] = acc
    print("%d | %.2f" % (k, acc))
    log("%d | %.2f" % (k, acc))


print('Singular acc')
print(float(sing_hit)/float(sing_tot))
print('Plural! ')
print(float(plur_hit)/float(plur_tot))

# print(dis_acc)
log('Accuracy by attractors')
print('Accuracy by attractors')
for k in dif_hit.keys():
    v = dif_hit[k]
    acc = v * 1./dif_tot[k]
    print("Attractors:",k,sep=" ")
    #print("Attractors: ",k)
    print("Accuracy:", acc, ", No. of sentences:",dif_tot[k], sep=" ")
    #print("Accuracy: ", acc, ", No. of sentences: ",dif_tot[k])
    log(k)
    log("{}, {}".format(acc, dif_tot[k]))
    dif_acc[k] = acc
    
#### saving interveners x attractors dictionary as pickle for plotting
if type(list(dif_acc)[0]) is tuple:
	with open(opt.rnn_arch + '_intdiff_acc.pickle', 'wb') as handle:
		pkl.dump(dif_acc, handle)

stats = {'distance': dis_acc, 'attractors': dif_acc}
torch.save(stats, opt.output)

log("Now doing perplexity analysis")
print("Now doing perplexity analysis")
valid = Dataset(opt.input, opt.dict, opt.batch_size, task='lm')
eval_model(model, valid)

log("Done!!")
print("Done!!")
