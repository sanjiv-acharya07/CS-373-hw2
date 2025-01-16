import matplotlib.pyplot as plt
import numpy as np
from preprocessing import load_titanic_data
from data_split import *

class NaiveBayesClassifier:

    def __init__(self, alpha):
        """
        Input:
            alpha - Integer parameter for Laplacian smoothing, typically set to 1
        """
        self.alpha = alpha

    def compute_class_probability(self, y_train):
        """
        Input: 
            y_train: Numpy array of class labels corresponding to the train data
        Return: 
            class_probabilities: dictionary where each key is a class label and 
                corresponding value is the probability of that label in y_train
        """
        
        class_probabilities = {}

        ## >>> YOUR CODE HERE >>>
        classes = np.unique(y_train)
        n = len(y_train)
        C = len(classes)
        for c in classes:
            class_probabilities[c] = (np.array(y_train == c).sum() + self.alpha) / (n + C*self.alpha)
        
        ## <<< END OF YOUR CODE <<<
        
        return class_probabilities
        
    def compute_feature_probability(self, Xj_train, y_train):
        """
        Input:
            Xj_train: a 1D array of strings with the values of a given feature 
                X_j for all data points
            y_train: a 1D array of strings with the class labels of all data 
                points
        Return:
            feature_probabilities: a dictionary whose entry (v, c) has the 
                computed probability of observing value 'v' among examples of
                class 'c', that is, P(X_j = c | Y = c).
                Note: v and c must be strings, the stored value must be float.
        """

        feature_probabilities = {}

        ## >>> YOUR CODE HERE >>>
        value_counts = {}  
        class_counts = {}  
        for i in range(len(Xj_train)):
            value = Xj_train[i]
            classs = y_train[i]            
            if classs not in value_counts:
                value_counts[classs] = {}
                class_counts[classs] = 0            
            if value not in value_counts[classs]:
                value_counts[classs][value] = 0            
            value_counts[classs][value] += 1
            class_counts[classs] += 1        
        unique = set(Xj_train)
        length = len(unique)        
        for classs in value_counts:
            for value in unique:                
                count = value_counts[classs].get(value, 0) + self.alpha                
                feature_probabilities[(value, classs)] = count / (class_counts[classs] + self.alpha*length)        
        ## <<< END OF YOUR CODE <<<
        return feature_probabilities


    def fit(self, X_train, y_train):
        """
        Fit Naive Bayes Classifier to the given data.

        This function computes all the necessary probability tables and stores
        them as dictionaries in the class.

        Input:
            X_train: a 2D numpy array, with string values, corresponding to the 
                pre-processed dataset
            y_train: a 1D numpy array, with string values, corresponding to the 
                pre-processed dataset

        Return:
            None
        """
        n, d = X_train.shape
        self.d = d

        # store the class labels in a list, with a fixed order
        self.class_labels = np.array(list(set(y_train)))


        self.class_probs = self.compute_class_probability(y_train)
        self.feature_probs = []
        for j in range(d):
            Xj = X_train[:, j]
            self.feature_probs.append(self.compute_feature_probability(Xj, y_train))

        return

    def predict_probabilities(self, X_test):
        """
        Input: X_test - 2D numpy array corresponding to the X for the test data
        Return: 
            probs - 2D numpy array with predicted probability for all classes, 
                for all test data points
        Objective: For the test data, compute posterior probabilities
        """
        probs = np.zeros((len(X_test), len(self.class_labels)))
        ## >>> YOUR CODE HERE >>>
        
        
        classes = self.class_labels
        test_size = len(X_test)
        
        for i in range(test_size):
            posterior_prob = {c: 0 for c in classes}
            for c in classes:
                posterior_prob[c] = self.class_probs[c]
                
                for j in range(self.d):
                    x = X_test[i, j]
                    dict1 = self.feature_probs[j]
                    l = dict1.get((x,c))
                    if l is not None:
                        posterior_prob[c] *= float(l)
                    
                probs[i, c] = posterior_prob[c]
        
        ## <<< END OF YOUR CODE <<<
        return probs

        
    def predict(self, probs):
        """Get predicted label from a matrix of posterior probabilities
        
        Input:
            probs: 2D numpy array with predicted probabilities
        Return:
            y_pred: 1D numpy array with predicted class labels (strings), based
                on the probabilities provided
        """
        
        
        ## >>> YOUR CODE HERE >>>
        max_indices = np.argmax(probs, axis=1)
        y_pred = self.class_labels[max_indices]
        
        ## <<< END OF YOUR CODE <<<
        
        return y_pred
        

        
    def evaluate(self, y_test, probs):
        """
        Compute the 0-1 loss and squared loss for the predictions

        Input: 
            y_test: true labels of test data
            probs: predicted probabilities from `predict_proba`
        Return:
            0-1 loss ( See homework pdf for
                their mathematical definition)
        """
        ## >>> YOUR CODE HERE >>>
        n = len(y_test)
        x = self.predict(probs)        
        zero_one_loss = np.sum(x != y_test) / n
        
        ## <<< END OF YOUR CODE <<<
        
        return zero_one_loss

"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""
import os

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    X, y = load_titanic_data(os.path.join(
        os.path.dirname(__file__), 'dataset/titanic.csv'))
    X_train, X_valid, y_train, y_valid = my_train_valid_split(
        X, y, 0.2, random_state=42)

    # Initialize and train a Naive Bayes classifier
    print('\n\n-------------Fitting NBC-------------\n')
    alpha = 1
    nbc = NaiveBayesClassifier(alpha)
    nbc.fit(X_train, y_train)
    print("Class Probabilities P_y:\n\n", {key: round(value, 2) for key, value in nbc.class_probs.items()})
    print('*'*40)
    print("\nFeature probabilites P_x_given_y:\n")
    cols=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    for i,col in enumerate(cols):
        feat_probs_i={key: round(value, 2) for key, value in nbc.feature_probs[i].items()}
        print(f'{col} : {feat_probs_i}\n')
    print('\tDone fitting NBC.')

    print('\n\n-------------Naive Bayes Performace-------------\n')
    probs = nbc.predict_probabilities(X_train)
    nbc.evaluate(y_train, probs)
    p_train = nbc.predict(probs)
    print('Train Accuracy: ',np.mean(y_train == p_train))

    probs = nbc.predict_probabilities(X_valid)
    nbc.evaluate(y_valid, probs)
    p_valid = nbc.predict(probs)
    print('Validation Accuracy: ',np.mean(y_valid == p_valid))

    print('\n\nDone.')
    

   


  






    

    

    
    
    

    

