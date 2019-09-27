from k_folds import k_folds

# Takes list of tuples containing instantiated models (not fitted) and name, and corpusData and corpusLabels lists and the vectorizer fitted to the vocabulary
# Returns a dict containing the k folds average accuracies for each model 
def getModelAccuracies(models, corpusData, corpusLabels, vectorizer, k):
    modelAccuracies = {}

    #For each model
    for model in models:
        model_name = model[1]

        #get the kfolds average accuracy
        k_folds_acc = k_folds(k,model[0],corpusData, corpusLabels, vectorizer)

        #put it in the dict
        modelAccuracies[model_name] = k_folds_acc

        
    return modelAccuracies


