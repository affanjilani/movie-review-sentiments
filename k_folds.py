from preprocess import partition, vectorizeData
from classifiers import classifier
from sklearn.base import clone

# Perform k folds cross validation on a given label
# k are the number of folds, model is the model to be used (instantiated) and corpus data and corpus labels are preprocessed lists
# vectorizer is already fitted on the entire vocabulary and will be used for creating the matrices
def k_folds(k, model, corpusData, corpusLabels, vectorizer):
    accuracySum = 0

    #Get partitions
    dataPartitions, labelPartitions = partition(corpusData=corpusData,corpusLabels=corpusLabels,partType='k-cross', k=5)

    #For each validation partition
    for vali_partition in range(k):
        #clone the model
        model_for_k = clone(model)

        #transform validation partition into counts
        validation_data = dataPartitions[vali_partition]
        validation_data = vectorizeData(validation_data,vectorizer)

        validation_labels = labelPartitions[vali_partition]

        #Use the remaining partitions as the training corpus
        training_data=[]
        training_labels=[]

        for train_partition in range(k):

            #don't add validation partition
            if train_partition == vali_partition:
                continue

            training_data += dataPartitions[train_partition]
            training_labels += labelPartitions[train_partition]
       
        # transform training_data into matrix of counts
        training_data = vectorizeData(training_data,vectorizer)

        #fit the model, predict and get accuracy
        accuracy_fold_k = classifier(model_for_k, training_data, training_labels, validation_data, validation_labels)

        accuracySum+=accuracy_fold_k['accuracy']

    #compute mean accuracy of model across folds
    averageAccuracy = accuracySum/k

    # return model accuracy
    return averageAccuracy