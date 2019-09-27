from sklearn.naive_bayes import MultinomialNB
from preprocess import readSentences, processFiles, vectorizeData, partition
import numpy as np

# Computes accuracy using prediction values and true values
def evaluate_acc(prediction, true):
    sum = 0

    #if we get true positive or true negative we add it to the sum
    for y, yhat in zip(prediction,true):
        sum = sum+1 if y == yhat else sum
    
    accuracy = sum/prediction.shape[0]

    return accuracy

corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg')

#partition data
testData, testLabels, validationData, validationLabels = partition(corpusData=corpusData, corpusLabels=corpusLabels,type='simple', testRatio=0.9)

#vectorize the testData and validation data
testData = vectorizeData(testData,vectorizer)
validationData = vectorizeData(validationData,vectorizer)

#try naive bayes
clf = MultinomialNB().fit(testData, testLabels)

# predict
predicted = clf.predict(validationData)

#accuracy
print(evaluate_acc(predicted,validationLabels))


