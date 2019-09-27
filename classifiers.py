import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# We try 4 classifiers to see which one gives best results

# Returns the different measures from training from a set and validating from another
def classifier(model, training_data, training_labels, validation_data, validation_labels):
    measures={}

    #fit a multinomial naive bayes
    clf = model.fit(training_data, training_labels)

    # predict
    predicted = clf.predict(validation_data)

    #add accuracy to measures
    measures['accuracy']=evaluate_acc(predicted,validation_labels)

    #add confusion matrix to measures
    measures['confusionMatrix']=confusion_matrix(validation_labels,predicted)

    return measures

def randomClassifier(training_data, training_labels, validation_data, validation_labels):
    measures={}

    predictedArray = np.random.randint(2, size=len(validation_labels))

    measures['accuracy']=evaluate_acc(predictedArray, validation_labels)

    return measures





# Evaluates the accuracy of a prediction vs the true labels
def evaluate_acc(prediction, true):
    sum = 0

    # if we get true positive or true negative we add it to the sum
    for y, yhat in zip(prediction, true):
        sum = sum+1 if y == yhat else sum

    accuracy = sum/prediction.shape[0]

    return accuracy
