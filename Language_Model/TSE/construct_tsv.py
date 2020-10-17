import os
import pickle 
import sys 
import pandas as pd


if len(sys.argv) != 2:
	print("Usage: python construct_tsv.py [template_dir]")
	sys.exit(1)


input_folder  =  sys.argv[1]

deps={'type':[], 'sentence':[], 't_verb':[], 'f_verb':[], 'verb_index':[]}
for f in os.listdir(input_folder):
    filename=str(f)
    print(type(filename))
    with open(os.path.join(filename, f), 'rb') as f:
        c = pickle.load(f)
    for key in c.keys():
        count=0
        for i in range(len(c[key])):
            count+=1
            deps['type'].append(str(filename)+"_"+key)
            sent1 =c[key][i][0].split()
            sent2 =c[key][i][1].split()
            for j in range(len(sent1)):
                if not sent1[j]==sent2[j]:
                    deps['verb_index'].append(j+1)
                    deps['t_verb'].append(sent1[j])
                    deps['f_verb'].append(sent2[j])
                    separator = ' '
                    deps['sentence'].append(separator.join(sent1))   
                    break


df = pd.DataFrame.from_dict(deps)

df.to_csv('tse.tsv', sep='\t', index=False)