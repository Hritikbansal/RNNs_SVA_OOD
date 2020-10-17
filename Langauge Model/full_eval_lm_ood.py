import torch
import layers
import argparse
import pickle as pkl
from text.utils import gen_inflect_from_vocab, deps_from_tsv
from itertools import zip_longest
from collections import Counter
import torch.nn.functional as F
from text.dataset import Dataset
import torch.nn as nn
import csv
import math
import pandas as pd 


dependency_fields = ['type',   'sentence' ,   't_verb' , 'f_verb' , 'verb_index']


def deps_to_tsv(deps, outfile):
    writer = csv.writer(open(outfile, 'w', encoding='utf-8'), delimiter='\t')
    writer.writerow(dependency_fields)
    for dep in deps:
        writer.writerow([dep[key] for key in dependency_fields])


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

inflect_verb, _ = gen_inflect_from_vocab('data/wiki.vocab')

# read data here
deps = deps_from_tsv(opt.input)
dis_hit = Counter()
dis_tot = Counter()
dif_hit = Counter()
dif_tot = Counter()
incorrect_deps=[]

bidx = 0
for i in range(0, len(deps), opt.batch_size):
    mb = []
    bidx += 1
    if bidx % 100 == 0:
        print('process {:5d} / {:d}'.format(i, len(deps)))
        log('process {:5d} / {:d}'.format(i, len(deps)))
    rejected=0
    for dep in deps[i: i+opt.batch_size]:
        t_verb = dep['t_verb']
        f_verb = dep['f_verb']
        type_ =  dep['type']
        t_verb = w2idx.get(t_verb, 1)
        if t_verb == 1:
            rejected+=1
            continue
        f_verb = w2idx.get(f_verb, 1)
        if f_verb == 1:
            rejected+=1
            continue
        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        ws = ['<bos>'] + tokens[:v]
        ws = [w2idx.get(w, 1) for w in ws]
        mb += [(ws, t_verb, f_verb, type_, dep)]
    print(rejected)
    mb.sort(key=lambda t: len(t[0]), reverse=True)
    xs = [x[0] for x in mb]
    tv = [x[1] for x in mb]
    fv = [x[2] for x in mb]
    xs = padding(xs, len(xs[0])).to(device) # Left padding! 
    scores, _ = model(xs, True)
    b = [i for i in range(len(tv))]
    true_scores = scores[b, tv]  # advance indexing
    fake_scores = scores[b, fv]
    corrects = true_scores.gt(fake_scores).view(-1).tolist()
    for i, v in enumerate(corrects):
        dx = mb[i][3]
        dis_tot[dx] += 1
        dis_hit[dx] += v
        if v==0:
            incorrect_deps.append(mb[i][-1])

deps_to_tsv(incorrect_deps, 'incorr_TSE.tsv')

dis_acc = {}
print(dis_tot)
log('Accuracy by distance')
print(dis_tot)
log('Accuracy by distance')
for k in sorted(dis_hit.keys()):
    v = dis_hit[k]
    acc = v / dis_tot[k]
    dis_acc[k] = acc
    print("{} | %.2f" % (k, acc))
    log("{} | %.2f" % (k, acc))

dictt={'Type':[], 'examples':[], 'Accuracy':[]}

for k in sorted(dis_hit.keys()):
    v = dis_hit[k]
    acc = v / dis_tot[k]
    dictt['Type'].append(k)
    dictt['examples'].append(dis_tot[k])
    dictt['Accuracy'].append(acc)

pd.DataFrame.from_dict(dictt).to_csv('result.tsv', sep='\t', index=False)


stats = {'type': dis_acc}
torch.save(stats, opt.output)

log("Done!!")
print("Done!!")
