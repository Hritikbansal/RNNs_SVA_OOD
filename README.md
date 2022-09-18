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

## Dataset

- The data used in this paper is taken Linzen et al. 2016 [paper](https://arxiv.org/abs/1611.01368). This can either be downloaded from [here](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz) or alternatively, this can also be downloaded using the script ``` download_data.sh``` (mentioned in the Language Model folder) using ```./download_data.sh```. 
- For TSE data, please look Language model folder. 
- For splitting and pre-processing the dataset, follow the following steps within the Language Model folder.
- python split_data.py -input "<path to data>/agr_50_mostcommon_10K.tsv" -out_folder expr_data -prop_train 0.90 --domain_adaptation  --compare_DA --K 0

  This outputs the training file for Selective setting and the test file. The other settings should have the no. of training sentences as in this one. In our case, this number was 103360:

- python split_data.py -input "<path to data>/agr_50_mostcommon_10K.tsv" -out_folder expr_data -prop_train 0.90 --train_size 103360
- python split_data.py -input "<path to data>/agr_50_mostcommon_10K.tsv" -out_folder expr_data -prop_train 0.90 --train_size 103360 --intermediate
- python split_data.py -input "<path to data>/agr_50_mostcommon_10K.tsv" -out_folder expr_data -prop_train 0.90 --train_size 103360 --domain_adaptation2


- Build Vocab: 
python build_dictionary.py expr_data


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
