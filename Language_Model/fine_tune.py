import argparse
import torch
import torch.nn as nn
import layers
import opts
from text.dataset import Dataset
import time
import math

def log(message):
    with open('logs/' + 'output.txt', 'a') as file:
        file.write(str(message) + '\n')


parser = argparse.ArgumentParser(description='Language Model')

parser.add_argument('-checkpoint', required=True,
                    help='file saved trained parameters')
parser.add_argument('-train', required=True, default='train.txt', type=str,
                    help='train file, one sentence per line.')
parser.add_argument('--rnn_arch', type=str, default='LSTM', 
                    help="Type of RNN architcture to be chosen from LSTM, GRU, ONLSTM, DRNN")
parser.add_argument('--activation', default='tanh', type=str, 
                    help='if to run DECAY RNN')
parser.add_argument('-hidden_size', required=True, default=650, type=int,
                    help='hidden unit size')
parser.add_argument('-dict', required=True, default='vocab.pkl',
                    help='vocabulary file.')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.preprocess_opts(parser)

opt = parser.parse_args()
# for grid search
opt.inner_size = 2 * opt.word_vec_size
opt.head_size = opt.word_vec_size // opt.num_heads

print(opt)
print('-' * 42)
log(opt)
log('-' * 42)

checkpoint = torch.load(opt.checkpoint)
print('| reconstruct network')
log('| reconstruct network')



torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)


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

def train(opt, checkpoint):
    print('| build data iterators')
    log('| build data iterators')
    train = Dataset(opt.train, opt.dict, opt.batch_size, task='lm', no_unks=True)

    print('| build model')

    log('| build model')
    if opt.n_words < 0:
        opt.n_words = len(train.dict)

    print('| vocab size %d' % opt.n_words)
    log('| build criterion')

    print('| vocab size %d' % opt.n_words)
    log('| build criterion')
    crit = build_crit(opt.n_words)

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
    print(len(train))
    log('| load parameters')
    model.load_state_dict(checkpoint['params'])

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for eidx in range(opt.epochs):
        model.train()
        tot_loss = 0
        n_words = 0
        train.shuffle()
        num_batches = len(train)
        ud_start = time.time()
        for i in range(len(train)):
            print(i)
            optimizer.zero_grad()
            x, y = prepare_batch(train[i])
            log_prob, _ = model(x)
            loss = crit(log_prob, y)
            nx = y.data.ne(0).int().sum().item()
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opt.max_grad_norm)

            optimizer.step()
            tot_loss += loss.item()
            n_words += nx

        print('Saving fine tuned model!!')
        checkpoint = {'params': model.state_dict(),
                      'opt': opt,
                      'best_valid_ppl': 0}
        torch.save(checkpoint, opt.save_model)

train(opt, checkpoint)
