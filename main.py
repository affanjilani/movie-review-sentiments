from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from preprocess import readSentences, processFiles, vectorizeData, partition
from classifiers import classifier, randomClassifier
import numpy as np
from preprocess import stem, lemmatize

corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType='lemma')

#partition data
testData, testLabels, validationData, validationLabels = partition(corpusData=corpusData, corpusLabels=corpusLabels,partType='simple', testRatio=0.9)

#vectorize the testData and validation data
testData = vectorizeData(testData,vectorizer)
validationData = vectorizeData(validationData,vectorizer)

bayes_measures = classifier(MultinomialNB(),testData,testLabels,validationData, validationLabels)
logistic_measures = classifier(LogisticRegression(solver='lbfgs', max_iter=500),testData,testLabels,validationData,validationLabels)
svm_measures = classifier(svm.LinearSVC(), testData, testLabels, validationData, validationLabels)
random_measures = randomClassifier(testData, testLabels, validationData, validationLabels)

print(bayes_measures, logistic_measures, svm_measures, random_measures)



