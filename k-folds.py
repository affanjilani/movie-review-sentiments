# Perform k folds cross validation on a given label
# k are the number of folds, model is the model to be used (instantiated) and corpus data and corpus labels are preprocessed lists
# vectorizer is already fitted on the entire vocabulary and will be used for creating the matrices
def k_folds(k, model, corpusData, corpusLabels, vectorizer):
    #Get partitions

    #Transform corpus data

    #For each validation partition

        #Use the remaining partitions as the training corpus

        #fit the model using this training corpus

        #predict and get accuracy

    #compute mean accuracy of model across folds

    # return model accuracy