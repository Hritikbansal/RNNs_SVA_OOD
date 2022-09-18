import argparse
import torch
import torch.nn as nn
import layers
import opts
from text.dataset import Dataset
import time
import math
import numpy as np

#torch.cuda.empty_cache() 

def log(message):
    with open('logs/' + 'output.txt', 'a') as file:
        file.write(str(message) + '\n')


parser = argparse.ArgumentParser(description='Language Model')

parser.add_argument('-train', required=True, default='train.txt', type=str,
                    help='train file, one sentence per line.')
#parser.add_argument('-valid', required=True, default='valid.txt', type=str,
 #                   help='validation file.')
parser.add_argument('--rnn_arch', type=str, default='LSTM', 
                    help="Type of RNN architcture to be chosen from LSTM, GRU, ONLSTM, DRNN")
parser.add_argument('-num_units', required=False, default=10, type=int,
                    help='hidden unit size')
parser.add_argument('-k', required=False, default=5, type=int,
                    help='hidden unit size')
parser.add_argument('--activation', default='tanh', type=str, 
                    help='if to run DECAY RNN')
parser.add_argument('-hidden_size', required=True, default=650, type=int,
                    help='hidden unit size')
# dictionaries
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


torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)


def prepare_xy(mb):
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
        x, y = prepare_xy(valid[i])
        log_prob, _ = model(x)
        nll = crit(log_prob, y)
        valid_nlls.append(nll.item())
        n_words += y.ne(0).int().sum().item()
    model.train()
    nll = torch.FloatTensor(valid_nlls).sum().item() / n_words
    return math.exp(nll)


def train(opt):
    print('| build data iterators')
    log('| build data iterators')
    print('| training data iterators')
    train = Dataset(opt.train, opt.dict, opt.batch_size, task='lm', dtype="train") ##
    print('| validation data iterators')
    valid = Dataset(opt.train, opt.dict, opt.batch_size, task='lm', dtype="valid") ##
    #print(valid)
    #print(len(valid.data))
    #print(len(valid._data))

    print('| building model')

    log('| build model')
    
    if opt.n_words < 0:
        opt.n_words = len(train.dict)

    print('| vocab size %d' % opt.n_words)
    #log('| build criterion')

    #print('| vocab size %d' % opt.n_words)
    log('| build criterion')
    crit = build_crit(opt.n_words)

    if opt.arch == 'rnn':
        if opt.rnn_arch =='DECAY':
            log('| build DECAY LM {}'.format(opt.activation))
            print('| build DECAY LM {}'.format(opt.activation))
        else:
            log('| build {} LM'.format(opt.rnn_arch))
            print('| build {} LM '.format(opt.rnn_arch))

        model = layers.RNNLM(opt.word_vec_size, opt.hidden_size, opt.n_words, opt.layers, 
                             opt.dropout, rnn_arch=opt.rnn_arch, activation = opt.activation, tied=opt.tied)
    elif opt.arch =='RIM':
        print("Building RIM architecture!!!!")
        model = layers.RIM_model(opt.word_vec_size, opt.hidden_size, opt.n_words, opt.layers, opt.num_units, opt.rnn_arch, opt.k,
                             opt.dropout, tied=opt.tied)
    else:
        print('| build Transformer')

        log('| build Transformer')
        model = layers.Transformer(opt.word_vec_size, opt.n_words,
                                   opt.num_heads, opt.head_size,
                                   opt.layers, opt.inner_size,
                                   opt.dropout, tied=opt.tied)
    print(model)

    log(model)

    model = model.to(device)
    # eval(model, valid, crit)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    best_valid_ppl = 1e10
    min_lr = opt.lr * math.pow(0.5, 5)
    for eidx in range(opt.epochs):
        model.train()
        tot_loss = 0
        n_words = 0
        
        # Preparing lists of lists as batches
        train.prep_batch()
        # batches of data stored in lists of lists: self._data
        #print(len(train._data[0]))
        
        
        num_batches = len(train)
        ud_start = time.time()
        for i in range(len(train)):
            optimizer.zero_grad()
            x, y = prepare_xy(train[i])
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
            if i % opt.report_every == 0 and i > 0:
                ud = time.time() - ud_start
                args = [eidx, i, num_batches, math.exp(tot_loss/n_words), best_valid_ppl, 
                        opt.report_every/ud]
                print("| Epoch {:2d} | {:d} / {:d} | ppl {:.3f} | best_valid_ppl {:.3f} "
                      "| speed {:.1f} b/s".format(*args))

                log("| Epoch {:2d} | {:d} / {:d} | ppl {:.3f} | best_valid_ppl {:.3f} "
                      "| speed {:.1f} b/s".format(*args))
                ud_start = time.time()

        print('| Evaluate')

        log('| Evaluate')

        model.eval()
        valid_ppl = eval(model, valid, crit)
        print('| Epoch {:2d} \t  | valid ppl {:.3f} \t | best valid ppl {:.3f}'
              .format(eidx, valid_ppl, best_valid_ppl))

        log('| Epoch {:2d} \t  | valid ppl {:.3f} \t | best valid ppl {:.3f}'
              .format(eidx, valid_ppl, best_valid_ppl))

        if valid_ppl <= best_valid_ppl:
            print('| Save checkpoint: %s | Valid ppl: %.3f' %
                  (opt.save_model, valid_ppl))

            log('| Save checkpoint: %s | Valid ppl: %.3f' %
                  (opt.save_model, valid_ppl))

            checkpoint = {'params': model.state_dict(),
                          'opt': opt,
                          'best_valid_ppl': valid_ppl}
            torch.save(checkpoint, opt.save_model)
            best_valid_ppl = valid_ppl
        else:
            opt.lr = opt.lr * 0.5
            if opt.lr < min_lr:
                print('reach minimum learning rate!')
                log('reach minimum learning rate!')

                exit()
            print('decay learning rate %f' % opt.lr)
            log('decay learning rate %f' % opt.lr)

            for group in optimizer.param_groups:
                group['lr'] = opt.lr


train(opt)
