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
def processFiles(positive, negative, processingType=None, partitions=0):
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

    vectorizer = CountVectorizer()

    #Learn the whole vocabulary
    vectorizer.fit(fullCorpus)

    #return set
    return  (corpusData, corpusLabels, vectorizer)

# Takes in the corpus data to count and returns the sparse matrix containing the counts
def vectorizeData(corpusData,vectorizer):
    # Creates nxm matrix and puts in the counts for each word in vocabulary
    # n is the number of samples
    # m is the features count
    dataCounts = vectorizer.transform(corpusData)

    return  dataCounts


#Reads in sentences from designated file and returns tuple  
def readSentences(filename):
    sentences = tuple(open(os.path.join('./rt-polaritydata/',filename)))

    return sentences

# Function that partitions data accordingly
# Simple simply creates a validation and test set based on the testRatio, k is ignored, returns (testData, testLabels, validationData, validationLabels)
# k-cross type ignores testRatio and returns a dict with the k partitions as (data, label) tuples
def partition(corpusData, corpusLabels, type, testRatio=0.9, k=5):
    if type == 'simple':
        numInTest = int(round(len(corpusData) * testRatio))

        testData = corpusData[0:numInTest]
        testLabels = corpusLabels[0:numInTest]

        validationData = corpusData[numInTest:]
        validationLabels = corpusLabels[numInTest:]

        return (testData,testLabels,validationData,validationLabels)
    elif type == 'k-cross':
        return None
    else:
        return None
