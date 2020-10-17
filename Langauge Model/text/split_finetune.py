# This code just splits the fine tune sentences, that are basically from the TSE corpus. 

import sys
import os
import errno
import random
from text.utils import deps_from_tsv
import argparse 

parser = argparse.ArgumentParser(description='Data split for parsing!!')
# for reproducibility
parser.add_argument('-input', required=True, type=str,
                    help='input tsv')

parser.add_argument('-out_folder', required=True, type=str,
                    help='proportion of training examples!')

parser.add_argument('-prop_train', required=True, default=0.1, type=float,
                    help='proportion of training examples!')
args = parser.parse_args()
random.seed(42)

dependency_fields = ['type', 'sentence', 't_verb', 'f_verb', 'verb_index']


def deps_to_tsv(deps, outfile):
    writer = csv.writer(open(outfile, 'w', encoding='utf-8'), delimiter='\t')
    writer.writerow(dependency_fields)
    for dep in deps:
        writer.writerow([dep[key] for key in dependency_fields])

prop_train = args.prop_train  # proportion of the data used for training


def prepare(fname, expr_dir):
    print('| read in the data')
    data = deps_from_tsv(fname)
    print('| shuffling')
    random.seed(42)
    random.shuffle(data)
    n_train = int(len(data) * prop_train)
    train = data[:n_train]
    test = data[n_train:]  # To keep the testing examples same. 

    try:
        os.mkdir(expr_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    print('| splitting')
    print('{} size {}'.format('Train', len(train)))
    print('{} size {}'.format('Test', len(test)))
    deps_to_tsv(train, os.path.join(expr_dir, 'train.tsv'))
    deps_to_tsv(test, os.path.join(expr_dir, 'test.tsv'))
    print('| done!')




if __name__ == '__main__':
    prepare(args.input, args.out_folder)

