# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:48:26 2022

@author: Varad Srivastava
"""

import pickle
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from matplotlib.ticker import PercentFormatter
import csv
import pandas as pd

"""SET DIRECTORY TO MODELS TOP LEVEL FOLDER"""
model_save_dir = "F:/STUDY/MS Cognitive Science IITD/Sem 3/Cognitive Science Project I - HSD 621/final_models"

def plot_var(train_task, train_setting, model_name):
    print(train_task, train_setting, model_name, sep=" ")
    if train_task == "Binary Classifier":
        with open(op.join(model_save_dir, train_task, train_setting, model_name, 
                          model_name + '_fullGram_intdiff_acc.pickle'), 'rb') as handle:
            data = pickle.load(handle)
            
    elif train_task == "Language Model":
        if model_name == "DECAY":
            model_name = "DRNN"
        with open(op.join(model_save_dir, train_task, train_setting, model_name, 
                          model_name + '_intdiff_acc.pickle'), 'rb') as handle:
            data = pickle.load(handle)
        
    print("Loaded Data")
    
    print(data)
    
    #FOR 0 attractor sentences
    x = [0,1,2,3,">3"]
    #sort the y list according to the i[0]
    if train_task == "Binary Classifier":
        a = 0
        pre_y = [(0,a),(1,a),(2,a),(3,a),(">3",a)]
        y = [data[i][0] for i in pre_y]
        #y = [data[i][0] for i in data if i[1]==0]
    elif train_task == "Language Model":
        #pre_y = [i for i in data if i[1]==0]
        a = 0
        pre_y = [(0,a),(1,a),(2,a),(3,a),(">3",a)]
        y = [data[i] for i in pre_y]
    plt.plot(x, y, label = "0 Attractor", marker="o")
    
    #FOR 1 attractor sentences
    x = [1,2,3,">3"]
    if train_task == "Binary Classifier":
        a = 1
        pre_y = [(1,a),(2,a),(3,a),(">3",a)]
        y = [data[i][0] for i in pre_y]
        #y = [data[i][0] for i in data if i[1]==1]
    elif train_task == "Language Model":
        #y = [data[i] for i in data if i[1]==1]
        a = 1
        pre_y = [(1,a),(2,a),(3,a),(">3",a)]
        y = [data[i] for i in pre_y]
    plt.plot(x, y, label = "1 Attractors", marker="o")
    
    #FOR 2 attractor sentences
    x = [2,3,">3"]
    if train_task == "Binary Classifier":
        a = 2
        pre_y = [(2,a),(3,a),(">3",a)]
        y = [data[i][0] for i in pre_y]
        #y = [data[i][0] for i in data if i[1]==2]
    elif train_task == "Language Model":
        #y = [data[i] for i in data if i[1]==2]
        a = 2
        pre_y = [(2,a),(3,a),(">3",a)]
        y = [data[i] for i in pre_y]
    plt.plot(x, y, label = "2 Attractors", marker="o")
    
    #FOR 3 attractor sentences
    x = [3,">3"]
    if train_task == "Binary Classifier":
        a =3
        pre_y = [(3,a),(">3",a)]
        y = [data[i][0] for i in pre_y]
        #y = [data[i][0] for i in data if i[1]==3]
    elif train_task == "Language Model":
        #y = [data[i] for i in data if i[1]==3]
        a =3
        pre_y = [(3,a),(">3",a)]
        y = [data[i] for i in pre_y]
    plt.plot(x, y, label = "3 Attractors", marker="o")
    
    #FOR >3 attractor sentences
    x = [">3"]
    if train_task == "Binary Classifier":
        a = ">3"
        pre_y = [(">3",a)]
        y = [data[i][0] for i in pre_y]
        #y = [data[i][0] for i in data if i[1]==">3"]
    elif train_task == "Language Model":
        #y = [data[i] for i in data if i[1]==">3"]
        a = ">3"
        pre_y = [(">3",a)]
        y = [data[i] for i in pre_y]
    plt.plot(x, y, label = ">3 Attractors", marker="o")
    
    plt.ylim(0.60, 1.00)
    
    if train_task == "Language Model":
        if model_name == "DRNN":
            plt.ylim(0.20,1.00)
        else:
            plt.ylim(0.60,1.00) #0.40 for decay RNN, for others: 0.60
        
    plt.title(train_task+": "+train_setting+" Setting "+model_name)
    plt.xlabel("No. of intervening nouns")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()
    

def deps_from_tsv(infile, limit=None):
    res = []
    for i, d in enumerate(csv.DictReader(open(infile, encoding='utf-8'), delimiter='\t')):
        if limit is not None and i >= limit:
            break
        res.append({x: int(y) if y.isdigit() else y for x, y in d.items()})
    return res


def plot_bydist(train_task, train_setting):
    
    """
    Investigating whether distance plays a role in the errors
    by checking variability in distance in error sentences
    """
    
    print(train_task, train_setting, sep=" ")
    
    
    if train_task == "Language Model":
        model_name = "DRNN"
    else:
        model_name = "DECAY"
            
    err1 = pd.read_csv(op.join(model_save_dir, train_task, train_setting, "LSTM",'incorr.tsv'),sep='\t')
    err2 = pd.read_csv(op.join(model_save_dir, train_task, train_setting, "ONLSTM",'incorr.tsv'),sep='\t')
    err3 = pd.read_csv(op.join(model_save_dir, train_task, train_setting, "GRU",'incorr.tsv'),sep='\t')
    err4 = pd.read_csv(op.join(model_save_dir, train_task, train_setting, model_name,'incorr.tsv'),sep='\t')
                          
    
    
    
    for i,data in enumerate([err1, err2, err3, err4]):
        
                
        dist = data["distance"]

        plt.hist(dist, bins=5, weights=np.ones(len(dist)) / len(dist), range=(0,10))
        plt.legend(loc='upper right')
        plt.xlabel("Distance between verb and noun")
        plt.ylabel("% of sentences")
        
        if i==0:
            model_name = "LSTM"
        elif i==1:
            model_name = "ONLSTM"
        elif i==2:
            model_name = "GRU"
        elif i==3:
            model_name = "DECAY RNN"
        
        plt.title(train_task+": "+train_setting+" Setting "+model_name+"\n Variation in distance in error sentences")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()

    
if __name__ == "__main__":
    
    print("Analysis Program for Accuracy variations over interveners x attractors")
    train_taskn = int(input("Input type of train task (1: BC,2: LM): "))
    if train_taskn==1:
        train_task = "Binary Classifier"
    elif train_taskn==2:
        train_task = "Language Model"
        
    train_settingn = int(input("Input type of training setting (1: Natural/2: Selective/3: Selective2 /4: Intermediate): "))
    if train_settingn==1:
        train_setting = "Natural"
    elif train_settingn==2:
        train_setting = "Selective"
    elif train_settingn==3:
        train_setting = "Selective 2"
    elif train_settingn==4:
        train_setting = "Intermediate"
        
    model_namen = int(input("Input type of model (1: LSTM/2: ONLSTM/3: GRU/4: DECAY RNN): "))
    if model_namen==1:
        model_name = "LSTM"
    elif model_namen==2:
        model_name = "ONLSTM"
    elif model_namen==3:
        model_name = "GRU"
    elif model_namen==4:
        model_name = "DECAY"
        
    plot_var(train_task, train_setting, model_name)
    
    #plot_bydist(train_task, train_setting)