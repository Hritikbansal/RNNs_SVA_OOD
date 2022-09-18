import pickle as pkl
import torch
from itertools import zip_longest
import random
import csv
from . import constants
from . import utils
import torch.nn.functional as F 


class Dataset(object):
    def __init__(self, tsv_file, vocab_file, batch_size=32, task='vp', no_unks=False, pos=False, pos_file=None, dtype=None):
        self.batch_size = batch_size
        self.pos_dict = None
        if pos:
            with open(pos_file, 'rb') as f:
                self.pos_dict = pkl.load(f)

        with open(vocab_file, 'rb') as f:
            self.dict = pkl.load(f)
            
        if dtype=="train": ##
            deps = utils.deps_from_tsv(tsv_file)
            lim = int(0.95*len(deps))
            deps = deps[:lim]
            print("Training size:"+str(len(deps)))
        elif dtype=="valid": ##
            deps = utils.deps_from_tsv(tsv_file)
            lim = int(0.95*len(deps))
            deps = deps[lim:]   
            print("Validation size:"+str(len(deps)))   
        else:
            deps = utils.deps_from_tsv(tsv_file)  
            print("test size:"+str(len(deps)))   
            
        self.no_unks=no_unks
        self.task = task
        if task == 'vp':
            self.task_vp(deps)
        elif task == 'pos':
            self.task_pos(deps)
        else:
            
            self.task_lm(deps, no_unks)
            # preprocessed sentences added in self.data
            
        self.prep_batch()
        # batches of data stored in lists of lists: self._data

    def padding(self, input_, max_len):
        padding_mask = [(max_len - len(x),0) for x in input_]
        for i in range(len(input_)):
            input_[i] =  F.pad(torch.LongTensor(input_[i]), padding_mask[i])

        return torch.stack(input_).t()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        assert index < self.num_batches
        return self._data[index]

    def prep_batch(self):
        if self.task == 'vp':
            self.shuffle_vp()
        elif self.task == 'pos':
            self.shuffle_pos()
        else:
            self.prep_batch_lm()
            
    def task_lm(self, deps, no_unks=False):
        print(" Preprocessing sentences (word to index)")
        self.data = []
        for dep in deps:
            xs = dep['sentence'].split()
            xs = [self.dict.get(x, constants.unk_idx) for x in xs]
            if constants.unk_idx in xs and no_unks:
                continue
            xs = [constants.bos_idx] + xs + [constants.eos_idx]
            self.data += [xs]

    def prep_batch_lm(self):
        #random.shuffle(self.data)
        print("Preparing batch for LM training")
        self._data = []
        
        for i in range(0, len(self.data), self.batch_size):
            mb = self.data[i: i+self.batch_size]
            mb.sort(key=len, reverse=True)
            max_len = len(mb[0])
            mb =  self.padding(mb, max_len)  # left padding! 
            self._data += [mb]
            #print(len(mb))
        #print(self._data[0][0]))

        self.num_batches = len(self._data)

    def shuffle_pos(self):
        random.shuffle(self.data)
        self._data = []
        for i in range(0, len(self.data), self.batch_size):
            mb = self.data[i: i+self.batch_size]
            mb.sort(key= lambda x : len(x[0]), reverse=True)
            max_len = len(mb[0][0])
            x, pos = zip(*mb)
            x =  list(x)
            pos = list(pos)
            x =  self.padding(x, max_len)  # left padding!
            pos  =  self.padding(pos, max_len) # left pad the pos as well!  
            self._data += [(x, pos)]

        self.num_batches = len(self._data)

    def shuffle_vp(self):
        random.shuffle(self.data)
        self._data = []
        for i in range(0, len(self.data), self.batch_size):
            mb = self.data[i: i+self.batch_size]
            mb.sort(key=lambda x: len(x[0]), reverse=True)
            mbx = [x[0] for x in mb]
            mby = [x[1] for x in mb]
            mbx = torch.LongTensor(
                list(zip_longest(*mbx, fillvalue=0)))
            mby = torch.FloatTensor(mby)
            self._data += [(mbx, mby)]

        self.num_batches = len(self._data)


    def task_pos(self, deps, no_unks=False):
        self.data = []
        for dep in deps:
            v = int(dep['verb_index']) - 1
            xs = dep['sentence'].split()[:v+1]
            pos = dep['pos_sentence'].split()[:v+1]
            xs = [self.dict.get(x, constants.unk_idx) for x in xs]
            pos = [self.pos_dict.get(x, constants.unk_idx) for x in pos]
            if constants.unk_idx in xs and no_unks:
                continue
            xs = [constants.bos_idx] + xs + [constants.eos_idx]
            pos = [constants.bos_idx] + pos + [constants.eos_idx]
            self.data += [(xs, pos)]


    def task_vp(self, deps):
        self.data = []
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        for dep in deps:
            v = int(dep['verb_index']) - 1
            x = dep['sentence'].split()[:v]
            y = self.class_to_code[dep['verb_pos']]
            x = [self.dict.get(w, constants.unk_idx) for w in x]
            self.data += [(x, y)]

