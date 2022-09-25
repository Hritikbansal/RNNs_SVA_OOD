# Binary Classification Objective

Going by [Linzen 2016 et al.](https://arxiv.org/pdf/1611.01368.pdf) experiments, this repo supports:
* Number Prediction Task 
* Grammaticality Judgement - Our paper only discusses this experiment.  

Network classes are implemented in pytorch.  

## Models

We consider 4 RNN models in our work.

* LSTM
* GRU
* [ONLSTM](https://arxiv.org/abs/1810.09536) - Ordered Neuron LSTM
* [DRNN](https://arxiv.org/abs/2005.08199) - Decay RNN 

## Data

```
python run.py --train --fullGram --augment_train --augment_test --save_data
```

```
python run.py --train --fullGram --augment_train --augment_test --train_file train_sel.tsv --domain_adaption --save_data
```

```
python run.py --train --fullGram --augment_train --augment_test --train_file train_inter.tsv --intermediate --save_data
```

```
python run.py --train --fullGram --augment_train --augment_test --train_file train_sel2.tsv --domain_adaption2 --save_data
```

## Usage

* If you want to train the models from scratch on Grammaticality Judgement task.
```
python run.py --model <RNN_ARCH> --train --fullGram (--<training setting>)
```

1. RNN_ARCH is one from - {'LSTM', 'DECAY', 'GRU', 'ONLSTM'}
2. <training setting> defines the training setting that is to be used, is one from - {domain_adaption, inter, domain_adaption2}
NOTE: This argument is not required for Natural setting. And "domain_adaption" and "domain_adaption2" refer to Selective and Selective 2 setting, while "inter" refers to Intermediate setting.

* If you want to test the trained models on Grammaticality Judgement task.
```
python run.py --model <RNN_ARCH> --fullGram --test_demarcated --validation_size 0
```

The above two experiments can be conducted through the code we shared for our different paper which had similar experiments - [repo](https://github.com/bhattg/Decay-RNN-ACL-SRW2020)


* Generally, the trained models will give aggregated accuracy rather than an distribution of accuracy over number of intervening nouns and number of attractors in a sentence. To get the accuracy over this joint distribution, you can ```--demarcate``` functionality while getting testing numbers.
```
python run.py --model <RNN_ARCH> --fullGram --test_demarcated --validation_size 0
```

* In our case ```run.py``` has ```load_data = True``` because we had saved the shuffled data in a folder, and were reusing it again during training rather than having to create the data from scratch again. We would recommend you to first run the code with ```load_data = False``` and save the constructed data somewhere.

Details of all the arguments is present in ```run.py```. 


## Disclaimer
* You can safely ignore ```--act_attention``` and ```--verb_embedding``` arguments for now as they were not used for the experiments conducted in the paper.
* We had performed new experiments pertaining to POS tagging as well, but those won't be needed to replicate results from the paper. They are just additional codes which have been integrated here in case you want to try new experiments.
* Please raise issues in case you are getting some unexpected behaviour or in case something seems to be missing, we would be happy to help.

