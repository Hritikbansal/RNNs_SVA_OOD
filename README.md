# RNN Subject Verb Agreement Out of Distribution Generalization Analysis

This is an official pytorch implementation for the experiments described in the paper - [Can RNNs trained on harder subject-verb agreement instances still
perform well on easier ones?](https://arxiv.org/pdf/2010.04976.pdf)

You can use anaconda (python library manager) or pip. 
We used anaconda, and downloaded all the dependencies in an anaconda environment.

## Requirements
- python 3
- pytorch >= 1.1
- numpy
- inflect
- statsmodels
- pandas
- matplotlib

## Data

- The data used in this paper is taken Linzen et al. 2016 [paper](https://arxiv.org/abs/1611.01368). This can either be downloaded from [here](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz) or alternatively, this can also be downloaded using the script ``` download_data.sh``` (mentioned in the Language Model folder) using ```./download_data.sh```. 
- For TSE data, please look Language model folder. 


- For splitting and pre-processing the dataset, execute the following snippets within the Language Model folder. Last 10% of the dataset was reserved as the testing set which remains identical across all settings. Training and validation sets were sampled from the initial 90% sentences in the dataset.

```
python split_data.py -input "<location to agr_50_mostcommon_10K.tsv>" -out_folder expr_data -prop_train 0.90 --domain_adaptation  --compare_DA --K 0
```

This snippet will generate the training file for Selective setting and the test file. note the size of the training set, and the validation set, we need the same size for other settings. In our case, this number was 103360.

```
python split_data.py -input "<location to agr_50_mostcommon_10K.tsv>" -out_folder expr_data -prop_train 0.90 --train_size 103360
```

```
python split_data.py -input "<location to agr_50_mostcommon_10K.tsv>" -out_folder expr_data -prop_train 0.90 --train_size 103360 --intermediate
```

```
python split_data.py -input "<location to agr_50_mostcommon_10K.tsv>" -out_folder expr_data -prop_train 0.90 --train_size 103360 --domain_adaptation2
```


Create dictionary
```
$ python build_dictionary.py expr_data
```
This script will write a `vocab.pkl` file to `expr_data`

## Accuracy Variation Plot and Qualitative Analyses

Use the script acc_variation_plot.py to plot accuracies variations over interveners x attractors. 
Do make sure:
1. To set your own "model_save_dir" at the beginning of the script.
2. "full_eval_lm.py" (in case of LM) and "decay_rnn_model.py" (in case of BC), upon testing, save "Interveners x attractors acc dictionary" as pickle file, whose moniker is of the form '<>_intdiff_acc.pickle'. These need to be in appropriate directories, following structure of: model_save_dir -> train_task -> train_setting -> model_name

For qualitative analyses, use the corresponding script "qualitative_analysis.py" in the following way. 
Choose <model_name> from ["lstm", "onlstm", "gru", "drnn"]
Do make sure:
1. To set your own "model_save_dir" at the beginning of the script.
2. The files containing incorrect prediction sentences "incorr.tsv" are in directory structure like: model_save_dir -> train_task -> train_setting -> model_name
```
python qualitative_analysis.py --model <model_name>
```

## Experiments

Broadly speaking, we train our models on two set of objectives - Grammaticality Judgement and Language Modeling. 
Hence, we divide our repo into two subparts solely dedicated to these objectives separately.

If you find our work useful, then please consider citing us using:
```
@article{https://doi.org/10.7275/5bnr-wc78,
  doi = {10.7275/5BNR-WC78},
  url = {https://scholarworks.umass.edu/scil/vol4/iss1/38/},
  author = {Bansal,  Hritik and Bhatt,  Gantavya and Agarwal,  Sumeet},
  title = {Can RNNs trained on harder subject-verb agreement instances still perform well on easier ones?},
  publisher = {University of Massachusetts Amherst},
  year = {2021}
}
```
