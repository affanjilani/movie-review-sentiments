import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# Reads positive and negative file and returns array of dicts representing the features vector as well as array of labels
# Label is 1 for positive, 0 for negative
# type: Type of further processing
#   None    : regular unigram of words
#   "lemma" : perform lemmatization of the words
#   "stem"  : perform stemming of the words
def processFiles(positive, negative, processingType=None):
    # Read in files as sentence
    positiveCorpus = readSentences(positive)
    negativeCorpus = readSentences(negative)

    fullCorpus = positiveCorpus + negativeCorpus

    #create the labels array, we know the first half of the corpus is positive(1) and second half negative(0)
    corpusLabels = []

    for i in range(int(len(fullCorpus)/2)):
        corpusLabels.append(1)
    
    for i in range(int(len(fullCorpus)/2)):
        corpusLabels.append(0)

    #shuffle the sentences before counting
    dataAndLabels = zip(fullCorpus,corpusLabels)

    dataAndLabels = list(dataAndLabels)

    np.random.shuffle(dataAndLabels)

    # Redivide sentences and their positivity/negativity
    corpusData = []
    corpusLabels = []
    for x,y in dataAndLabels:
        corpusData.append(x)
        corpusLabels.append(y)

    # Create vector of unigrams for whole corpus
    vect = CountVectorizer()

    # Creates nxm matrix and puts in the counts for each word in vocabulary
    # n is the number of samples
    # m is the features count
    dataCounts = vect.fit_transform(corpusData)



    #return set
    return  (dataCounts, corpusLabels)

#Reads in sentences from designated file and returns tuple  
def readSentences(filename):
    sentences = tuple(open(os.path.join('./rt-polaritydata/',filename)))

    return sentences