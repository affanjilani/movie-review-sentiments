from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from preprocess import readSentences, processFiles, vectorizeData, partition, removeStopWords
from classifiers import classifier, randomClassifier
import numpy as np
from preprocess import stem, lemmatize
from k_folds import k_folds
from model_selection import getModelAccuracies

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType=None)

# print(len(vectorizer.get_feature_names()))

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType=None, min_df=0.001)

# print(len(vectorizer.get_feature_names()))

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType='stop')

# print(len(vectorizer.get_feature_names()))

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType='stop', min_df=0.001)

# print(len(vectorizer.get_feature_names()))

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType='lancaster')

# print(len(vectorizer.get_feature_names()))

# corpusData, corpusLabels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType='lancaster', min_df=0.001)

# print(len(vectorizer.get_feature_names()))

#partition data
#testData, testLabels = partition(corpusData=corpusData, corpusLabels=corpusLabels,partType='k-cross', k=5)


#vectorize the testData and validation data
# testData = vectorizeData(testData,vectorizer)
# validationData = vectorizeData(validationData,vectorizer)

# bayes_measures = classifier(MultinomialNB(),testData,testLabels,validationData, validationLabels)
# logistic_measures = classifier(LogisticRegression(solver='lbfgs', max_iter=500),testData,testLabels,validationData,validationLabels)
# svm_measures = classifier(svm.LinearSVC(), testData, testLabels, validationData, validationLabels)
# random_measures = randomClassifier(testData, testLabels, validationData, validationLabels)

# print(bayes_measures, logistic_measures, svm_measures, random_measures)
# names = ['Naive Bayes', 'Logistic Regression', 'Linear SVM']
# models = [MultinomialNB(),LogisticRegression(solver='lbfgs', max_iter=500),svm.LinearSVC()]
# models = zip(models,names)

# accuracies = getModelAccuracies(models,corpusData,corpusLabels,vectorizer, 5)

# print(accuracies)


