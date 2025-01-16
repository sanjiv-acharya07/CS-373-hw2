from random import randrange, seed
import numpy as np
import pandas as pd

def load_titanic_data(path,flag=False):     
    """
        Read the data from titanic.csv. 
        Preprocess the data by doing the following:
        (i) Drop columns that are not useful for classification.
        (ii) Handle missing values.
        (iii) Quantize continuous features.
        (iv) Convert categorical features to numeric.

        Args:
            path: path to the dataset

        Returns:
            X: data
            y: labels
    """

    data = pd.read_csv(path)

    # (i) Drop columns that are not useful for classification (PassengerId, Name, Ticket, Cabin)
    ## >>> YOUR CODE HERE >>>
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    ## <<< END OF YOUR CODE <<<

    # (ii) Handle missing values
    # Fill missing 'Age' values with the median and missing 'Embarked' with a feature value "Unknown"
    ## >>> YOUR CODE HERE >>>
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna('Unknown')
    ## <<< END OF YOUR CODE <<<

    # (iii) Quantize continuous features
    # Quantize 'Age' into 3 categories ('young','middle-aged','old')
    # and 'Fare' into 4 categories ('low','medium','high','very high')
    ## >>> YOUR CODE HERE >>>
    data['Fare'] = pd.qcut(data['Fare'], q=4, labels=['low', 'medium', 'high', 'very high'])
    data['Age'] = pd.qcut(data['Age'].rank(method='first'), q=3, labels=['young', 'middle-age', 'old'])
    
    ## <<< END OF YOUR CODE <<<

    # (iv) Convert categorical features (Sex, Embarked, Pclass, Age, Fare) to numeric
    ## >>> YOUR CODE HERE >>>
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})
    data['Age'] = data['Age'].map({'young':0, 'middle-age':1, 'old':2})
    data['Fare'] = data['Fare'].map({'low': 0, 'medium': 1, 'high': 2, 'very high': 3})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2, 'Unknown': 3})
    
    
    ## <<< END OF YOUR CODE <<<

    if flag:
        print('\nFirst 5 rows of data after preprocessing:\n')
        print(data.head())
        print("="*40)

        # Loop through each column and print the feature name and unique values
        print('\nUnique values per feature:\n')
        for column in data.columns:
            unique_vals = data[column].nunique()
            print(f"Feature: {column} - Unique Values: {unique_vals}")
        print("="*40)

    X = data.drop(columns=['Survived']).values
    y = data['Survived'].values

    return X, y

load_titanic_data('dataset/titanic.csv',True)

