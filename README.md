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



## Experiments

Broadly speaking, we train our models on two set of objectives - Grammaticality Judgement and Language Modeling. 
Hence, we divide our repo into two subparts solely dedicated to these objectives separately.

If you find our work useful, then please consider citing us using:
```
@article{bansal2020trained,
author = {Hritik Bansal and Gantavya Bhatt and Sumeet Agarwal},
title = {Can RNNs trained on harder subject-verb agreement instances still perform well on easier ones?},
year = {2020},
journal = {arXiv:2010.04976},
}
```
