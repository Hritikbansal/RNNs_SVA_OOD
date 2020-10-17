import torch
import layers
import argparse
import pickle as pkl
from text.utils import gen_inflect_from_vocab, deps_from_tsv
from text import constants
from itertools import zip_longest
from collections import Counter
import torch.nn.functional as F
from text.dataset import Dataset
import torch.nn as nn
import math
import sys
import pandas as pd 

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

def give_sos_index(xs):
    # returns index of start of sequence token
    # input tensor of shape (time, bsz)
    time, bsz =  xs.shape 
    sos_idx=[]
    for b in range(bsz):
        for i in range(time):
            if xs[i, b].cpu().item() == constants.bos_idx:
                sos_idx.append(i)
                break
    return sos_idx

def get_logprobs(xs, out, t_verb, f_verb, true_sent, false_sent):
    # xs : input sentences of the shape (time, bsz)
    # out : output logpprobs of rnn: (time, bsz, vocab_dim)
    # t_verb: For each example, the corresponding t_verb index : list
    # f_verb: For each example, the corresponding false_verb index.: list 
    # returns the list (a batch technically) of tuples, where each tuple is log prob of sent and sent with f_verb per token. 
    sos_idx = give_sos_index(xs) #len bsz 
    Len =  out.shape[0]
    bsz =  xs.shape[1]
    log_prob_list=[]
    for b in range(bsz):
        corr_time =  [i for i in range(sos_idx[b], Len)]
        sent_root = xs[sos_idx[b]+1 :, b].cpu().tolist()  
        corr_sent = [*sent_root, t_verb[b]] # required noun idx
        incorr_sent = [*sent_root, f_verb[b]]
        assert len(corr_time) == len(corr_sent)
        logprobs_corr = out[corr_time, b, corr_sent]
        logprobs_incorr = out[corr_time, b, incorr_sent] # tensor of length same as the sequence length 
        assert len(logprobs_corr) ==  Len - sos_idx[b]
        assert len(logprobs_incorr) ==  Len - sos_idx[b]
        log_prob_list.append((logprobs_corr, logprobs_incorr, true_sent[b], false_sent[b]))

    return log_prob_list


def segregat_configWise(log_prob_list, type_):
    d={}
    for i in range(len(type_)):
        if type_[i] in d.keys():
            d[type_[i]].append(log_prob_list[i])
        else:
            d[type_[i]]=[log_prob_list[i]]
    return d

def add_entry(master_dict, slave_dict):
    # slave will update the master
    for keys in slave_dict.keys():
        if keys in master_dict.keys():
            master_dict[keys]+=slave_dict[keys]
        else:
            master_dict[keys]=[]+slave_dict[keys]
    return master_dict


parser = argparse.ArgumentParser(description='Evaluate LM')
parser.add_argument('-checkpoint', required=True,
                    help='file saved trained parameters')
parser.add_argument('-input', required=True, help='input file.')
parser.add_argument('-output', required=True, help='output for plotting.')
parser.add_argument('-batch_size', type=int, default=256,
                    help='batch size, set larger for speed.')
parser.add_argument('--rnn_arch', type=str, default='LSTM', 
                    help="Type of RNN architcture to be chosen from LSTM, GRU, ONLSTM, DRNN")
opt = parser.parse_args()

checkpoint = torch.load(opt.checkpoint)
saved_opt = checkpoint['opt']
saved_opt.input = opt.input
saved_opt.output = opt.output
saved_opt.rnn_arch =  opt.rnn_arch
saved_opt.batch_size = opt.batch_size
opt = saved_opt
print('| reconstruct network')
log('| reconstruct network')

if opt.arch == 'rnn':
    model = layers.RNNLM(opt.word_vec_size, opt.hidden_size, opt.n_words, opt.layers,
                        opt.dropout, rnn_arch=opt.rnn_arch, activation = opt.activation, tied=opt.tied)
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

# read data here
deps = deps_from_tsv(opt.input)
dis_hit = Counter()
dis_tot = Counter()
dif_hit = Counter()
dif_tot = Counter()

bidx = 0
predicted_probs={}

for i in range(0, len(deps), opt.batch_size):
    mb = []
    predicted_per_batch=[]
    bidx += 1
    if bidx % 100 == 0:
        print('process {:5d} / {:d}'.format(i, len(deps)))
        log('process {:5d} / {:d}'.format(i, len(deps)))
    for dep in deps[i: i+opt.batch_size]:
        t_verb = dep['t_verb']
        f_verb = dep['f_verb']
        _, type_ =  dep['type'].split(".pickle_")  # type_ is the configuration
        t_verb = w2idx.get(t_verb, 1)
        if t_verb == 1:
            continue
        f_verb = w2idx.get(f_verb, 1)
        if f_verb == 1:
            continue
        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        ws = ['<bos>'] + tokens[:v]
        ws = [w2idx.get(w, 1) for w in ws]
        mb += [(ws, t_verb, f_verb, type_, tokens[:v]+[dep['t_verb']], tokens[:v]+[dep['f_verb']])]
    mb.sort(key=lambda t: len(t[0]), reverse=True)
    xs = [x[0] for x in mb]
    tv = [x[1] for x in mb]
    fv = [x[2] for x in mb]
    xs = padding(xs, len(xs[0])).to(device) # Left padding! 
    type_ = [x[3] for x in mb]
    true_sent = [x[4] for x in mb]
    false_sent = [x[5] for x in mb]
    with torch.no_grad():
        scores, _ = model(xs, False) # scores will be of (time, bsz, vsz)

    demarcated = segregat_configWise(get_logprobs(xs, scores, tv, fv, true_sent, false_sent), type_)
    predicted_probs =  add_entry(predicted_probs, demarcated)


print("Dumping predicted probs")
with open('logprobs.pkl', 'wb') as f:
    pkl.dump(predicted_probs, f)

log("Done!!")
print("Done!!")
