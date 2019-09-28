# Movie Review Sentiment Model

## Goal
---
This is a small project to dip toes into the world of Machine Learning and NLP. It uses a dataset provided by the authors of `Pang and Lillian Lee, Seeing stars: Exploiting class relationships for sentiment categorization with
                        respect to rating scales, Proceedings of ACL 2005.`

## The Project
___
Essentially, we try a variety of different text preprocessing techniques and send this data to three kinds of models

*  Naive Bayes
*  Logistic Regression
*  Linear SVM

### Procedure
With the end goal being to have a script that would test how the 3 models perform on various processings of the data, I reached this goal through the following steps:

1.  Processing
    
    Done in the 
    [preprocess.py](preprocess.py) file. Here we define functions to read in the positive and negative data, create lists of data and labels, as well as preprocess the data. We also define a function that partitions the said data into either k partitions (for k-folds) or into a simple ratio of training and validation sets.

2.  Model accuracies using K-folds cross validation

    Defined a function `getModelAccuracies` in 
    [model_selection.py](model_selection.py) to take in a list of models and return the accuracies of those models in the form of dict, given data and the number of folds to perform.

    The folds algorithm is implemented in 
    [k_folds.py](k_folds.py)

### Running the experiment to obtain data
All of this comes together in the 
[experiments.py](experiments.py) file where the accuracies of all three models are printed based off of the type of text processing that was done on the data.
To run:
```bash
python experiments.py
```



