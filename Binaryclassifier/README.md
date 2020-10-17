# Binary Classification Objective

Going by [Linzen 2016 et al.](https://arxiv.org/pdf/1611.01368.pdf) experiments, this repo supports:
* Number Prediction Task 
* Grammaticality Judgement - Our paper only discusses this experiment.  

Network classes are implemented in pytorch.  

## Models

We consider 4 RNN models in our work.

* Vanilla RNN
* LSTM
* GRU
* [ONLSTM](https://arxiv.org/abs/1810.09536) - Ordered Neuron LSTM
* [DRNN](https://arxiv.org/abs/2005.08199) - Decay RNN 
* Other variants of DRNN like SDRNN and AbDRNN

## Usage

* Sample Usage 1 - If you want to train the models from scratch on Grammaticality Judgement task.
```
python run.py --model LSTM --train --fullGram
```

* Sample Usage 2 - If you want to test the trained models on Grammaticality Judgement task.
```
python run.py --model LSTM --fullGram 
```

The above two experiments can be conducted through the code we shared for our different paper which had similar experiments - [repo](https://github.com/bhattg/Decay-RNN-ACL-SRW2020)

* Sample Usage 3 - For domain adaption experiment - Training on harder sentences - sentences with attractor >= 1
```
python run.py --model LSTM --train --fullGram --domain_adaption
```

* Sample Usage 4 - In our paper, we perform augmentation to the training data which doubles the amount of training data.
```
python run.py --model LSTM --train --fullGram --augment_train <OPTIONAL --domain_adaption>
```

* Sample Usage 5 - Generally, the trained models will give aggregated accuracy rather than an distribution of accuracy over number of intervening nouns and number of attractors in a sentence. To get the accuracy over this joint distribution, you can ```--demarcate``` functionality while getting testing numbers.
```
python run.py --model LSTM --fullGram --test_demarcated
```

* In our case ```run.py``` has ```load_data = True``` because we had saved the shuffled data in a folder, and were reusing it again during training rather than having to create the data from scratch again. We would recommend you to first run the code with ```load_data = False``` and save the constructed data somewhere.

Details of all the arguments is present in ```run.py```. 


## Disclaimer
* You can safely ignore ```--act_attention``` and ```--verb_embedding``` arguments for now as they were not used for the experiments conducted in the paper.
* We had performed new experiments pertaining to POS tagging as well, but those won't be needed to replicate results from the paper. They are just additional codes which have been integrated here in case you want to try new experiments.
* Please raise issues in case you are getting some unexpected behaviour or in case something seems to be missing, we would be happy to help.

