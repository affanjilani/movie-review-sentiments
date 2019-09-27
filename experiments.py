# Script that runs experiments to find best models depending on preprocessing type
# Model selection is done by finding the best average accuracy of the model through k-folds cross validation using k =10
# For each preprocessing of the corpus, it then takes the best model and trains it using a 95, 10 split to produce a confusion matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from classifiers import classifier, randomClassifier
from preprocess import processFiles, partition, vectorizeData
from model_selection import getModelAccuracies

def compareModels():

    processing_types = [None,'porter','lemma', 'stop', 0.0001]

    classifier_objects = [MultinomialNB(), LogisticRegression(solver='lbfgs', max_iter=500), svm.LinearSVC()]

    classifier_names = ["Naive Bayes", "Logistic Regression", "Linear SVM"]



    # Create list of differently processed data
    processed_data = []
    for proc_type in processing_types:
        data_name = ""

        if proc_type == None:
            data_name = "No Processing"
        
        elif type(proc_type) == float:
            data_name = "Removing infrequent words at threshold " + str(proc_type)
            processedTuple = processFiles('rt-polarity.pos','rt-polarity.neg',processingType=None,min_df=proc_type)

            processed_data+=[(data_name,processedTuple)]
            continue
        
        else:
            data_name = "Corpus processing using " + str(proc_type)

        processedTuple = processFiles('rt-polarity.pos','rt-polarity.neg',processingType=proc_type,min_df=1)
        
        processed_data+=[(data_name,processedTuple)]

    # Now run the different classifiers on this data
    accuracies = []

    for dataName, data in processed_data:

        models = zip(classifier_objects,classifier_names)

        corpusData, corpusLabels, vectorizer = data

        accuracies_for_data = getModelAccuracies(models,corpusData,corpusLabels,vectorizer, 10)

        print("\n================== "+dataName+" ==================")
        for name, acc in accuracies_for_data.items():
            print(name+" Accuracy over 10 folds: "+str(acc))

        accuracies += [accuracies_for_data]

# Gives back a confusion matrix using the specified processing of corpus and the threshold frequency 
def createModelConfusionMatrix(model, proc_type, min_df=1, partitionRatio=0.95):
    data, labels, vectorizer = processFiles('rt-polarity.pos','rt-polarity.neg',processingType=proc_type,min_df=min_df)

    trainingData,trainingLabels,validationData,validationLabels = partition(data,labels,'simple',testRatio=partitionRatio)

    trainingData = vectorizeData(trainingData,vectorizer)
    validationData = vectorizeData(validationData, vectorizer)

    measures = classifier(model,trainingData,trainingLabels,validationData,validationLabels)

    
    print("\n================== "+"Accuracy"+" ==================")
    print(measures['accuracy'])
    print("\n================== "+"Confusion Matrix"+" ==================")
    print(measures['confusionMatrix'])


compareModels()

createModelConfusionMatrix(MultinomialNB(),None)


