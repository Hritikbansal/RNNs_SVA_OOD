import sys
import os
import errno
import random
from text.utils import deps_from_tsv, deps_to_tsv
import argparse 

parser = argparse.ArgumentParser(description='Data split for parsing!!')
# for reproducibility
parser.add_argument('-input', required=True, type=str,
                    help='input tsv')

parser.add_argument('-out_folder', required=True, type=str,
                    help='proportion of training examples!')

parser.add_argument('-prop_train', required=True, default=0.1, type=float,
                    help='proportion of training examples!')
parser.add_argument('--prop_val', default=0.05, type=float,
                    help='proportion of validation examples!')

parser.add_argument('--mixup_ratio', default=0.5, type=float,
                    help='proportion of mixup examples!')

parser.add_argument('--validation_size', default=7000, type=int,
                    help='size for testing the data. Default is 7k corresponding to the validation split.')

parser.add_argument('--train_size', default=100000000000000, type=int,
                    help='size for training the data. Default is 7k corresponding to the validation split.')

parser.add_argument('--test_size', default=7000, type=int,
                    help='size for testing the data. Default is 7k corresponding to the validation split.')

parser.add_argument('--domain_adaptation', action='store_true',
                    help='If doing domain adaptation testing!')

parser.add_argument('--compare_DA', action='store_true',
                    help='If preparing the representatinos for the graph!')

parser.add_argument('--K', default=1, type=int,
                    help='K value in the domain adaptation')

parser.add_argument('--n_intervening', action='store_true',
                    help='If doing domain adaptation n_intervening!')

parser.add_argument('--mixup', action='store_true', 
                    help='If want to do mixup!')

parser.add_argument('--i', default=0, type=int, help=
                    'n_intervening  for mixup')

parser.add_argument('--j', default=0, type=int, help=
                    'n_diff_intervening  for mixup')

args = parser.parse_args()

random.seed(42)

prop_train = args.prop_train  # proportion of the data used for training

def mixup(train, test, args, criteria):
    #mixup ratio : the amount of (i, j) to be taken as prop of count of min allowable attrctor value ( or K value)
    counter=0
    for dep in train:
        if dep[criteria] == args.K+1:
            counter+=1
    n_mix =int(args.mixup_ratio*counter)
    train_ind_list=[]
    test_ind_list =[]
    count=0
    for i in range(len(test)):
        if count==n_mix:
            break
        if (test[i]['n_intervening'], test[i]['n_diff_intervening']) == (args.i, args.j):
            count+=1
            train_ind_list.append(i)
        else:
            test_ind_list.append(i)
    for j in range(i, len(test)):
        test_ind_list.append(j)
    train =  train  + get_indices(test, train_ind_list)
    random.seed(42)
    random.shuffle(train)
    test =  get_indices(test, test_ind_list)
    return train, test

def get_indices(input_, ind_list):
    output=[]
    for ind in ind_list:
        output.append(input_[ind])
    return output

def prepare(fname, expr_dir):
    print('| read in the data')
    data = deps_from_tsv(fname)
    print('| shuffling')
    random.seed(42)
    random.shuffle(data)
    n_train = int(len(data) * prop_train)
    n_valid = int(len(data) * args.prop_val)
    train = data[:n_train]
    valid = data[n_train:n_train+n_valid]
    test = data[n_train+n_valid:]  # To keep the testing examples same. 

    valid=valid[:args.validation_size] # This is used in the ns case in our domain adaptation. 
    train=train[:args.train_size]
    try:
        os.mkdir(expr_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    print('| splitting')
    print('{} size {}'.format('Train', len(train)))
    print('{} size {}'.format('valid', len(valid)))
    print('{} size {}'.format('Test', len(test)))
    deps_to_tsv(train, os.path.join(expr_dir, 'train.tsv'))
    deps_to_tsv(valid, os.path.join(expr_dir, 'valid.tsv'))
    deps_to_tsv(test, os.path.join(expr_dir, 'test.tsv'))
    print('| done!')


# def prepare_DA(fname, expr_dir):
#     # This method is not used in the paper, so this can be safely ignored! 
#     print('| read in the data')
#     data = deps_from_tsv(fname)
#     print('| shuffling')
#     print("| Domain adaptation!")
#     random.seed(42)
#     random.shuffle(data)
#     criteria = 'n_intervening' if args.n_intervening else 'n_diff_intervening'
#     print(criteria)
#     n_train = int(len(data) * prop_train)

#     train_indices=[]
#     for i in range(n_train):
#         if len(train_indices)>=args.train_size:
#             break
#         if data[i][criteria] > args.K:
#             train_indices.append(i)

#     train= get_indices(data, train_indices) 
#     if args.train_size < len(train):
#         train=  train[:args.train_size]   # so that we can also control number of training examples. 


#     validation_indices =[]
#     for j in range(i, len(data)):  #see only in the remaining set of examples. 
#         if data[j][criteria] > args.K:
#             validation_indices.append(j)
#     valid =  get_indices(data, validation_indices)
#     n_valid =  int(0.5*len(valid))   # only use the half of the validation examples for the true validation and rest half with testing as the trend investigation. 

#     test_indices=[]
#     for i in range(len(data)):
#         if data[i][criteria] <= args.K:
#             test_indices.append(i)
#     test = get_indices(data, test_indices[:args.test_size])
#     test = test + valid[n_valid:]
#     valid = valid[:n_valid]

#     print('Premixup length!')
#     print('Length of training sentences is {}'.format(len(train)))
#     print('Length of validation sentences is {}'.format(len(validation_indices)))
#     print('Lenght of testing sentences is {}'.format(len(test_indices[:args.test_size])))

#     try:
#         os.mkdir(expr_dir)
#     except OSError as exc:
#         if exc.errno != errno.EEXIST:
#             raise
#         pass
#     if args.mixup:
#         print('| Mixup it is!')
#         train, test= mixup(train, test, args, criteria)

#     print('Length of training sentences is {}'.format(len(train)))
#     print('Length of validation sentences is {}'.format(len(valid)))
#     print('Lenght of testing sentences is {}'.format(len(test)))
#     print('| splitting')
#     deps_to_tsv(train, os.path.join(expr_dir, 'train.tsv'))
#     deps_to_tsv(valid, os.path.join(expr_dir, 'valid.tsv'))
#     deps_to_tsv(test, os.path.join(expr_dir, 'test.tsv'))
#     print('| done!')

def prepare_compare_DA(fname, expr_dir):
    # This will take domain adapted training examples till the point of prop_train
    # Then will take the validaiton and testing examples from the rest of the prop_train 
    # NOte that, the testing set will be the same across K value. 
    print('| read in the data')
    data = deps_from_tsv(fname)
    print('| shuffling')
    print("| Domain adaptation!")
    random.seed(42)
    random.shuffle(data)
    criteria = 'n_intervening' if args.n_intervening else 'n_diff_intervening'
    print(criteria)
    n_train = int(len(data) * prop_train)
    n_valid = int(len(data)*args.prop_val)

    train_indices=[]
    for i in range(n_train):
        if data[i][criteria] > args.K:
            train_indices.append(i)
    train= get_indices(data, train_indices) 

    validation_indices =[]
    for j in range(n_train, n_valid+n_train):  #see only in the remaining set of examples. 
        if data[j][criteria] > args.K:
            validation_indices.append(j)
    valid =  get_indices(data, validation_indices)
 
    test = data[n_train+n_valid:] # use all the last ones for the testing. 

    try:
        os.mkdir(expr_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('Length of training sentences is {}'.format(len(train)))
    print('Length of validation sentences is {}'.format(len(valid)))
    print('Lenght of testing sentences is {}'.format(len(test)))
    print('| splitting')
    deps_to_tsv(train, os.path.join(expr_dir, 'train.tsv'))
    deps_to_tsv(valid, os.path.join(expr_dir, 'valid.tsv'))
    deps_to_tsv(test, os.path.join(expr_dir, 'test.tsv'))
    print('| done!')


if __name__ == '__main__':
    if args.domain_adaptation:
        if args.compare_DA:
            prepare_compare_DA(args.input, args.out_folder)
        # else:
        #     prepare_DA(args.input, args.out_folder)     # This method is not used in the paper, so this can be safely ignored! 
    else:
        prepare(args.input, args.out_folder)

