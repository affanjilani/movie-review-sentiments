import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer


# Reads positive and negative file and returns array of dicts representing the features vector as well as array of labels
# Label is 1 for positive, 0 for negative
# type: Type of further processing
#   None         : regular unigram of words
#   "lemma"      : perform lemmatization of the words
#   "lancaster"  : perform stemming of the words using lancaster algorithm
#   "porter"     : perform stemming of the words using porter algorithm
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

    # Perform any stemming or lemmatization here
    if processingType == 'lemma':
        corpusData = lemmatize(corpusData)
    elif processingType == 'porter' or processingType == 'lancaster':
        corpusData = stem(corpusData, processingType)
    

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
def partition(corpusData, corpusLabels, partType, testRatio=0.9, k=5):
    if partType == 'simple':
        numInTest = int(round(len(corpusData) * testRatio))

        testData = corpusData[0:numInTest]
        testLabels = corpusLabels[0:numInTest]

        validationData = corpusData[numInTest:]
        validationLabels = corpusLabels[numInTest:]

        return (testData,testLabels,validationData,validationLabels)
    elif partType == 'k-cross':
        return None
    else:
        return None

# Stems a corpus according to either the porter stemmer or lancaster one
# takes in a corpus as a list of sentences and returns the stemmed version of those sentences in the same order
def stem(corpus, stemType='lancaster'):
    stemmedSentences = []

    stemmer = LancasterStemmer() if stemType=='lancaster' else PorterStemmer()

    #for each sentence in the corpus
    for sentence in corpus:

        #tokenize the sentence to get the words
        tokens = word_tokenize(sentence)
        
        stemmedSentence = ""
        
        # Go through each word and tokenize it
        for word in tokens:
            #stem each word and then add it to the stemmed sentence
            stemmedSentence += stemmer.stem(word) + " "

        # Once all words have been stemmed we have the stemmed version of the sentence and we append it to list
        stemmedSentences.append(stemmedSentence)
        
    # Return the stemmed sentences
    return stemmedSentences

# Lemmatizes a corpus consisting of a list of sentences
def lemmatize(corpus):
    # Create the lemmatizer object
    lemmatizer = WordNetLemmatizer()

    #For each sentence in the list
    lemmatizedSentences = []
    for sentence in corpus:
        lemmatizedSentence = ""

        #tokenize each sentence
        tokens = word_tokenize(sentence)

        #For each token we lemmatize and make the sentence again
        for word in tokens:
            lemmatizedSentence += lemmatizer.lemmatize(word) + " "

        lemmatizedSentences.append(lemmatizedSentence)

    return lemmatizedSentences



