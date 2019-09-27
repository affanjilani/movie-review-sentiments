import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')


# Reads positive and negative file and returns array of dicts representing the features vector as well as array of labels
# Label is 1 for positive, 0 for negative
# type: Type of further processing
#   None         : regular unigram of words
#   "lemma"      : perform lemmatization of the words
#   "lancaster"  : perform stemming of the words using lancaster algorithm
#   "porter"     : perform stemming of the words using porter algorithm 
#   "stop"       : remove stop words
def processFiles(positive, negative, processingType=None, min_df=1):
    # Read in files as sentence
    positiveCorpus = readSentences(positive)
    negativeCorpus = readSentences(negative)

    fullCorpus = positiveCorpus + negativeCorpus

    # Perform any stemming or lemmatization here
    if processingType == 'lemma':
        fullCorpus = lemmatize(fullCorpus)
    elif processingType == 'porter' or processingType == 'lancaster':
        fullCorpus = stem(fullCorpus, processingType)
    elif processingType == 'stop':
        fullCorpus = removeStopWords(fullCorpus)

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
    

    vectorizer = CountVectorizer(min_df=min_df)

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
# k-cross type ignores testRatio and returns a tuple (dataPartitions, labelPartitions) where dataPartitions is a list of a list of sentences
def partition(corpusData, corpusLabels, partType, testRatio=0.9, k=5):
    if partType == 'simple':
        numInTest = int(round(len(corpusData) * testRatio))

        testData = corpusData[0:numInTest]
        testLabels = corpusLabels[0:numInTest]

        validationData = corpusData[numInTest:]
        validationLabels = corpusLabels[numInTest:]

        return (testData,testLabels,validationData,validationLabels)
    elif partType == 'k-cross':
        dataPartitions = []
        labelsPartitions = []

        # Figure out how many elements in each partition if perfectly divisible by k
        num_elems_in_partitions = int(len(corpusLabels)/k)
        
        # Figure out how many elements are left, will be less than k
        left_over = int(len(corpusLabels) - (num_elems_in_partitions * k))

        end = 0
        # We partition the data
        for partition in range(k):

            # start at index we last stopped at
            start = end
            # depending on how many elements are left over, we add an extra element to the partition
            end = start + num_elems_in_partitions + 1 if partition < left_over else start+num_elems_in_partitions 

            # get the subset of data in this partition
            dataSubset = corpusData[start:end]

            # get the subset of labels in this partition
            labelSubset = corpusLabels[start:end]

            # Add the subsets to the right lists
            dataPartitions.append(dataSubset)
            labelsPartitions.append(labelSubset)


        return (dataPartitions, labelsPartitions)
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

# Takes in a list of sentences as the corpus and outputs a list of those sentences without the stop words
def removeStopWords(corpus):
    sentencesWithoutStops = []

    stop_words = set(stopwords.words('english'))

    # for each sentence to remove stop words 
    for sentence in corpus:
        sentenceWithoutStops = ""

        # tokenize it so we have only the words
        tokens = word_tokenize(sentence)

        # for each of the words
        for word in tokens:
            # If the word is not in the list of stop words add it to the sentence
            if word not in stop_words:
                sentenceWithoutStops += word + " "

        # Append the new sentence to the master list
        sentencesWithoutStops.append(sentenceWithoutStops)

    return sentencesWithoutStops




