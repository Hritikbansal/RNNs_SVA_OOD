# RNNs_SVA_OOD

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

- We used the annotated dataset provided by [Linzen et al. 2016](https://github.com/TalLinzen/rnn_agreement) which consists of sentences scrapped from Wikipedia.
- For Targeted Syntactic Evaluation, templates are available [here](https://drive.google.com/file/d/13Q_zUz5fZxYwGuo_-ZbS20HHUkZEHPzl/view) OR you can refer to [Marvin and Linzen 2018](https://github.com/BeckyMarvin/LM_syneval/tree/master/EMNLP2018/templates) repo.


## Experiments

Broadly speaking, we train our models on two set of objectives - Grammaticality Judgement and Language Modeling. 
Hence, we divide our repo into two subparts solely dedicated to these objectives separately.

