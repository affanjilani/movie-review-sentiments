import os
import nltk
import numpy as np

# Reads positive and negative file and returns array of dicts representing the features vector as well as array of labels
# Label is 1 for positive, 0 for negative
# type: Type of further processing
#   None    : regular unigram of words
#   "lemma" : perform lemmatization of the words
#   "stem"  : perform stemming of the words
def processFiles(filename1, filename2, processingType=None):
    data = ""
    # Read in file1 as sentence
    file = open(os.path.join('./rt-polaritydata',filename1),'r',encoding='ISO-8859-1')
    
    for sentence in file:
        data+=sentence

    file.close()

    # Read in file2 as sentence
    file = open(os.path.join('./rt-polaritydata',filename2),'r',encoding='ISO-8859-1')

    for sentence in file:
        data+=sentence

    file.close()

    # Tokenize the whole corpus
    full_unigram = nltk.word_tokenize(data)
    #get rid of the duplicates
    full_unigram = set(full_unigram)

    print(len(full_unigram))


    # TODO:Further process the words based on type inputted


    # Make 2D data array and 1D label array
    data_array, label_array = generateData(filename1, filename2, full_unigram, processingType)


    # Shuffle the data
    condensed = zip(data_array,label_array)

    np.random.shuffle(condensed)

    data_array = [item[0] for item in condensed]
    label_array = [item[1] for item in condensed]


    #return set
    return  (data_array, label_array)

# Generates the full feature vectors and labels using the full unigram set
def generateData(positiveFilename, negativeFilename, unigramSet, processingType=None):
    data_array = []
    label_array = []
    
    #Get the positive and negative sentences
    positiveSentences = readSentences(positiveFilename)
    negativeSentences = readSentences(negativeFilename)


    # TODO: perform any stemming or lemmatization here

    # Make the feature vectors for positive sentences
    for sentence in positiveSentences:
        # Get tokens
        tokens = nltk.word_tokenize(sentence)

        # Get unique unigrams
        unigrams = set(tokens)

        # Create big feature dict
        featureVector = {}
        for unigram in unigramSet:
            featureVector[unigram] = 0

        # Get count of unigrams in sentence and add the count to featureVector
        for unigram in unigrams:
            featureVector[unigram] = tokens.count(unigram)

        # Append to data array
        data_array.append(featureVector)

        # Add positive label to label array
        label_array.append(1)

    # Make the feature vectors for negative Sentences
    for sentence in negativeSentences:
        # Get tokens
        tokens = nltk.word_tokenize(sentence)

        # Get unique unigrams
        unigrams = set(tokens)

        # Create big feature dict
        featureVector = {}
        for unigram in unigramSet:
            featureVector[unigram] = 0

        # Get count of unigrams in sentence and add the count to featureVector
        for unigram in unigrams:
            featureVector[unigram] = tokens.count(unigram)

        # Append to data array
        data_array.append(featureVector)

        # Add positive label to label array
        label_array.append(0)

    return (data_array, label_array)

#Reads in sentences from designated file and returns tuple  
def readSentences(filename):
    sentences = tuple(open(os.path.join('./rt-polaritydata/',filename)))

    return sentences