# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:58:25 2022

@author: Varad Srivastava
"""

#import pickle
#import matplotlib.pyplot as plt
import os.path as op
import csv
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import argparse

"""SET DIRECTORY TO MODELS TOP LEVEL FOLDER""" #Can be made part of arg parser in future code iterations
model_save_dir = "F:/STUDY/MS Cognitive Science IITD/Sem 3/Cognitive Science Project I - HSD 621/final_models"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Choose model from: lstm, onlstm, gru, drnn")
args = parser.parse_args()

def search(sent,errf):
    
    for i in errf:
        if i["sentence"] == sent:
            return True
        
    return False
        

def deps_from_tsv(infile, limit=None):
    res = []
    for i, d in enumerate(csv.DictReader(open(infile, encoding='utf-8'), delimiter='\t')):
        if limit is not None and i >= limit:
            break
        res.append({x: int(y) if y.isdigit() else y for x, y in d.items()})
    return res

def error_analysis(train_task, model_name, train_setting1, train_setting2 ):
    print("===============================================================")
    print(train_task, model_name, train_setting1, train_setting2, sep=" ")
    print("===============================================================")
    
    if train_setting1 == train_setting2:
        print("ERROR: Make sure that the train settings to be compared are different.")
        sys.exit()
    
    #if train_task == "Language Model":
     #   model_name = "DRNN"
    #else:
     #   model_name = "DECAY"
        
    test = deps_from_tsv(op.join(model_save_dir, "expr_data/test.tsv"))
    
    err1 = deps_from_tsv(op.join(model_save_dir, train_task, train_setting1, model_name,'incorr.tsv'))

                          
    print("Errors in {} setting: {}".format(train_setting1, len(err1)))
    
    err2 = deps_from_tsv(op.join(model_save_dir, train_task, train_setting2, model_name,'incorr.tsv'))

    print("Errors in {} setting: {}".format(train_setting2, len(err2)))
    
    #common errors
    common_err = 0
    for j in err2:
        sent = j["sentence"]
        if search(sent,err1):
            common_err+=1
            
    print("Common errors: {}".format(common_err))
        
    #calculate the difference in errors across the two settings
    diff_set = []
    
    for j in err2:
        sent = j["sentence"]
        if search(sent,err1):
            continue
        diff_set.append(j)  #errors in err2 which do not occur in err1 (natural)

    
    print("No. of errors in {} setting which did not happen in {} setting -> {}".format(train_setting2,
                                                                                train_setting1, len(diff_set)))
    
    
    # for incorrect sentences with 0 intervening nouns
    diff_set00 = []
    
    for k in diff_set:
        if k["n_intervening"] == 0:
            diff_set00.append(k)
            
    
            
    print("No. of error sentences with 0 intervening nouns in {} setting which did not happen in {} setting -> {} \n".format(train_setting2,
                                                                                train_setting1, len(diff_set00)))

    return err1, err2, diff_set00
    #return test


def hyp_test1_propernoun(errors, set):
    """
    subject trailed by proper noun of different number are causing error
    """
    
    leng = len(errors)
    nouns_before = 0
    hyp1_errors = []
    
    
    for i in errors:
        noun_bflag=0
        pos = i["pos_sentence"].split(" ")
        #posi = pos[i["subj_index"]-1]
        subj_ind = i["subj_index"]-1
        subj_pos = i["subj_pos"]
        
        if subj_ind==0:
            continue
        
        else:
            for j in range(subj_ind):
                
                if pos[j] in ["NN", "NNS", "NNP", "NNPS"]:
                    noun_bflag=1
                    # or pos[j]=="NNP" or pos[j]=="NNPS" FOR PROPER NOUN
                    
                    if (pos[j] == "NN" or pos[j]=="NNP") and subj_pos=="NNS":
                        hyp1_errors.append(i)
                        break
                    
                    elif (pos[j] == "NNS" or pos[j]=="NNPS") and subj_pos=="NN":
                        hyp1_errors.append(i)
                        break
                        
            if noun_bflag:
                nouns_before+=1
                
                
    hyp1_percent = len(hyp1_errors)/leng
    
    print("No. of sentences in which atleast one noun appeared before the subject in {}: {}".format(set, nouns_before))
    print("No. of sentences in which any of these preceding noun was of different number than the subject in {}: {}".format(set, len(hyp1_errors)))
    print("% of sentences in which any of these preceding noun was of different number than the subject in {}: {}".format(set, hyp1_percent))

    return hyp1_errors, hyp1_percent



def hyp_test1(errors, set):
    """
    subject trailed by noun of different number are causing error
    """
    
    leng = len(errors)
    nouns_before = 0
    hyp1_errors = []
    
    for i in errors:
        noun_bflag=0
        pos = i["pos_sentence"].split(" ")
        #posi = pos[i["subj_index"]-1]
        subj_ind = i["subj_index"]-1
        subj_pos = i["subj_pos"]
        
        if subj_ind==0:
            continue
        
        else:
            for j in range(subj_ind):
                
                if pos[j] in ["NN", "NNS"]:
                    noun_bflag=1
                    # or pos[j]=="NNP" or pos[j]=="NNPS" FOR PROPER NOUN
                    
                    if pos[j] == "NN" and subj_pos=="NNS":
                        hyp1_errors.append(i)
                        break
                    
                    elif pos[j] == "NNS" and subj_pos=="NN":
                        hyp1_errors.append(i)
                        break
                        
            if noun_bflag:
                nouns_before+=1
                
                
    hyp1_percent = len(hyp1_errors)/leng
    
    print("No. of sentences in which atleast one noun appeared before the subject in {}: {}".format(set, nouns_before))
    print("No. of sentences in which atleast one of these preceding noun was of different number than the subject in {}: {}".format(set, len(hyp1_errors)))
    print("% of sentences in which atleast one of these preceding noun was of different number than the subject in {}: {}".format(set, hyp1_percent))

    return hyp1_errors, hyp1_percent


def hyp_test2(errors, set):
    """
    compound nouns: incorrect head of a compound noun identified
    """

    leng = len(errors)
    incor_head = 0
    hyp2_errors = []

    for i in errors:
        pos = i["pos_sentence"].split(" ")
        subj_ind = i["subj_index"] - 1
        subj_pos = i["subj_pos"]

        if subj_ind == 0:
            continue

        else:
            j = subj_ind - 1
            # while j!=-1 or pos[j] not in ["NN","NNS","JJ",""]:

            if subj_pos == "NN":

                if pos[j] in ["NNS", "JJ", "VBG"]:
                    incor_head += 1
                    hyp2_errors.append(i)


            elif subj_pos == "NNS":

                if pos[j] in ["NN", "JJ", "VBG"]:
                    incor_head += 1
                    hyp2_errors.append(i)

                # j-=1

    hyp2_percent = len(hyp2_errors) / leng

    print(
        "No. of sentences in which a noun of different number OR a JJ/VBG appeared before the subject in {}: {}".format(
            set, incor_head))
    print("% of sentences in which a noun of different number OR a JJ/VBG appeared before the subject in {}: {}".format(
        set, hyp2_percent))

    return hyp2_errors, hyp2_percent


def hyp_test3(errors, set):
    """
    singular vs plural subject errors?
    """
    plural = 0
    singul = 0
    leng = len(errors)

    for i in errors:
        subj_pos = i["subj_pos"]

        if subj_pos == "NNS":
            plural += 1
        elif subj_pos == "NN":
            singul += 1

    print("No. of incorrect sentences with plural subject in {}: {}".format(set, plural))
    print("% of incorrect sentences with plural subject in {}: {}".format(set, plural / leng))
    print("No. of incorrect sentences with singular subject in {}: {}".format(set, singul))
    print("% of incorrect sentences with singular subject in {}: {}".format(set, singul / leng))


def hyp_test4(errors):
    """
    verbs occuring in the very beginning and the end are throwing up errors
    """

    verb_inds = []
    subj_inds = []
    sen_len = []
    for i in errors:
        subj_inds.append(i['subj_index'])
        verb_inds.append(i['verb_index'])
        sen_len.append(len(i['sentence'].split(" ")))

    return subj_inds, verb_inds, sen_len


def hyp_test4a(errors, thres, set):
    """
    subjects occuring NOT in the very beginning and the end are throwing up errors
    """

    leng = len(errors)

    sen = []
    for i in errors:
        sen_len = len(i['sentence'].split(" "))
        ratio = i["subj_index"] / sen_len

        if ratio < thres[1] and ratio > thres[0]:
            sen.append(i)

    print("No. of such sentences without primacy/recency in {}: {}".format(set, len(sen)))
    print("% of such sentences without primacy/recency in {}: {}".format(set, len(sen) / leng))


def hyp_test5(errors, set):
    """
    subject trailed by noun - majority voting effect happening in errors (for number of verb)?
    """

    leng = len(errors)
    nouns_before = 0
    hyp5_errors = 0

    for i in errors:
        noun_bflag = 0
        sing_noun = 0
        plur_noun = 0
        pos = i["pos_sentence"].split(" ")
        # posi = pos[i["subj_index"]-1]
        subj_ind = i["subj_index"] - 1
        subj_pos = i["subj_pos"]
        verb_pos = i["verb_pos"]

        if subj_ind == 0:
            continue

        else:
            for j in range(subj_ind+1):

                if pos[j] in ["NN", "NNS", "NNP", "NNPS"]:
                    noun_bflag = 1
                    # or pos[j]=="NNP" or pos[j]=="NNPS" FOR PROPER NOUN

                    #if (pos[j] == "NN" or pos[j] == "NNP") and subj_pos == "NNS":
                    if pos[j] in ["NN", "NNP"]:
                        sing_noun+=1

                    elif pos[j] in ["NNS", "NNPS"]:
                        plur_noun+=1

            if noun_bflag:
                nouns_before += 1


        if (verb_pos=="VBP" and sing_noun>plur_noun) or (verb_pos=="VBZ" and sing_noun<plur_noun):
            # sentences among incorrect ones where majority voting would pass
            # their over representation would show that model struggles in these
            hyp5_errors+=1



    #hyp5_percent = len(hyp5_errors) / leng

    #print("No. of sentences in which atleast one noun appeared before the subject: {}".format(nouns_before))
    print("No. of sentences in which majority voting effect is observed in {}: {}".format(set, hyp5_errors))
    print("% of sentences in which majority voting effect is observed in {}: {}".format(set, hyp5_errors/len(errors)))

    #return hyp5_errors, hyp5_percent
    
if __name__ == "__main__":
    
    print("Qualitative analysis of errors")
    
    """
    train_taskn = int(input("Input type of train task (1: BC,2: LM): "))
    if train_taskn==1:
        train_task = "Binary Classifier"
    elif train_taskn==2:
        train_task = "Language Model"
        
    train_settingn1 = int(input("Input type of training setting 1 (1: Natural/2: Selective/3: Selective2 /4: Intermediate): "))
    if train_settingn1==1:
        train_setting1 = "Natural"
    elif train_settingn1==2:
        train_setting1 = "Selective"
    elif train_settingn1==3:
        train_setting1 = "Selective 2"
    elif train_settingn1==4:
        train_setting1 = "Intermediate"
        
    train_settingn2 = int(input("Input type of training setting 2 (1: Natural/2: Selective/3: Selective2 /4: Intermediate): "))
    if train_settingn2==1:
        train_setting2 = "Natural"
    elif train_settingn2==2:
        train_setting2 = "Selective"
    elif train_settingn2==3:
        train_setting2 = "Selective 2"
    elif train_settingn2==4:
        train_setting2 = "Intermediate"
        
    """
    train_task="Language Model"
    #model_name = "lstm"
    model_name = args.model
    train_setting1 = "Natural"
    train_setting2 = "Intermediate"
    err_nat, err_inter, errors_nat_inter = error_analysis(train_task, model_name, train_setting1, train_setting2)

    err_nat0 = []
    for e in err_nat:
        if e["n_intervening"] == 0:
            err_nat0.append(e)

    train_setting1 = "Natural"
    train_setting2 = "Selective"
    err_nat, err_sel, errors_nat_sel = error_analysis(train_task,model_name, train_setting1, train_setting2)
    
    test = deps_from_tsv(op.join(model_save_dir, "expr_data/test.tsv"))
    
    test0 = []
    for t in test:
        if t["n_intervening"] == 0:
            test0.append(t)


    # SAVING DIFF SET ERROR SENTENCES
    keys = errors_nat_sel[0].keys()
    with open('errors_lstm_nat_sel.csv', 'w', newline='', encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(errors_nat_sel)

    keys = errors_nat_inter[0].keys()
    with open('errors_lstm_nat_inter.csv', 'w', newline='', encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(errors_nat_inter)


    print("==================== HYPOTHESIS 1: SUBJECTS TRAILED BY NOUN OF DIFF NO ARE CAUSING ERROR ====================")
    print("# TEST SET: BASELINE")
    hyp1_errors_baseline, hyp1_percent_baseline = hyp_test1(test0, "Test")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN SELECTIVE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp1_errors_nat_sel, hyp1_percent_nat_sel = hyp_test1(errors_nat_sel, "Nat_Sel")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN INTERMEDIATE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp1_errors_nat_inter, hyp1_percent_nat_inter = hyp_test1(errors_nat_inter, "Nat_Inter")
    print("\n")

    print("# COMPARE AGAINST INDIVIDUAL ERROR SETS")
    hyp1_errors_nat0, hyp1_percent_nat0 = hyp_test1(err_nat0, "Nat0")
    hyp1_errors_nat, hyp1_percent_nat = hyp_test1(err_nat, "Nat")
    hyp1_errors_sel, hyp1_percent_sel = hyp_test1(err_sel, "Sel")
    hyp1_errors_inter, hyp1_percent_inter = hyp_test1(err_inter, "Inter")
    print("\n" * 2)



    print("==================== HYPOTHESIS 2: COMPOUND NOUNS ARE CAUSING ERROR ====================")
    print("# TEST SET: BASELINE")
    hyp2_errors_test, hyp2_percent_test = hyp_test2(test0, "Test")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN SELECTIVE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp2_errors_nat_sel, hyp2_percent_nat_sel = hyp_test2(errors_nat_sel, "Nat_Sel")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN INTERMEDIATE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp2_errors_nat_inter, hyp2_percent_nat_inter = hyp_test2(errors_nat_inter, "Nat_Inter")
    print("\n")

    print("# COMPARE AGAINST INDIVIDUAL ERROR SETS")
    hyp2_errors_nat0, hyp2_percent_nat0 = hyp_test2(err_nat0, "Nat0")
    hyp2_errors_nat, hyp2_percent_nat = hyp_test2(err_nat, "Nat")
    hyp2_errors_sel, hyp2_percent_sel = hyp_test2(err_sel, "Sel")
    hyp2_errors_inter, hyp2_percent_inter = hyp_test2(err_inter, "Inter")
    print("\n"*2)



    print("==================== HYPOTHESIS 3: SINGULAR VS PLURAL SUBJECTS ====================")
    print("# TEST SET: BASELINE")
    hyp_test3(test0, "Test")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN SELECTIVE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test3(errors_nat_sel, "Nat_Sel")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN INTERMEDIATE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test3(errors_nat_inter, "Nat_Inter")
    print("\n")

    print("# COMPARE AGAINST INDIVIDUAL ERROR SETS")
    hyp_test3(err_nat0, "Nat0")
    hyp_test3(err_nat, "Nat")
    hyp_test3(err_sel, "Sel")
    hyp_test3(err_inter, "Inter")
    print("\n" * 2)

    print("==================== HYPOTHESIS 4: RECENCY OR PRIMACY EFFECTS ====================")
    """
    In addition, our analyses revealed primacy and recency effects in the
    generalization patterns of LSTM-based models, showing that these models tend to perform well on
    the outer- and innermost parts of a center-embedded tree structure, but poorly on its middle levels.
    """

    ratio = [0.3, 0.7]

    print("# TEST SET: BASELINE")
    hyp_test4a(test0, ratio, "Test")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN SELECTIVE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test4a(errors_nat_sel, ratio, "Nat_Sel")
    print("\n")

    print(
        "# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN INTERMEDIATE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test4a(errors_nat_inter, ratio, "Nat_Inter")
    print("\n")

    print("# COMPARE AGAINST INDIVIDUAL ERROR SETS")
    hyp_test4a(err_nat0, ratio, "Nat0")
    hyp_test4a(err_nat, ratio, "Nat")
    hyp_test4a(err_sel, ratio, "Sel")
    hyp_test4a(err_inter, ratio, "Inter")
    print("\n" * 2)


    print("==================== HYPOTHESIS 5: MAJORITY VOTING EFFECT BY TRAILING NOUNS? ====================")
    print("# TEST SET: BASELINE")
    hyp_test5(test0, "Test")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN SELECTIVE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test5(errors_nat_sel, "Nat_Sel")
    print("\n")

    print("# FOR ERROR SENTENCES WITH 0 INTERVENING NOUNS IN INTERMEDIATE SETTING WHICH DIDNOT OCCUR IN NATURAL SETTING #")
    hyp_test5(errors_nat_inter, "Nat_Inter")
    print("\n")

    print("# COMPARE AGAINST INDIVIDUAL ERROR SETS")
    hyp_test5(err_nat0, "Nat0")
    hyp_test5(err_nat, "Nat")
    hyp_test5(err_sel, "Sel")
    hyp_test5(err_inter, "Inter")


    #future iteration: can output all this as dataframe