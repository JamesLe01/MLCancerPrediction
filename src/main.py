import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearnex import patch_sklearn
from MyLogisticRegression import MyLogisticRegression
from MySVM import MySVM
from MyDecisionTree import MyDecisionTree
from MyGaussianNaiveBayes import MyGaussianNaiveBayes


def load_data(file_name: str):
    df = pd.read_csv(file_name)
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'].replace(['B', 'M'], [0, 1], inplace=True)  # Benign = 0, Malicious = 1
    
    y = df['diagnosis'].to_numpy()
    X = df.drop('diagnosis', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, X_test, y_train, y_test):
    logistic_reg = MyLogisticRegression(X_train, X_test, y_train, y_test)
    accuracy = logistic_reg.train()
    print(f'Accuracy: {accuracy}')
    print('[[TN FP]\n [FN TP]]=')
    print(logistic_reg.get_confusion_matrix())
    
def svm(X_train, X_test, y_train, y_test):
    svm = MySVM(X_train, X_test, y_train, y_test)
    accuracy = svm.train()
    print(f'Accuracy: {accuracy}')
    print('[[TN FP]\n [FN TP]]=')
    print(svm.get_confusion_matrix())

def decision_tree(X_train, X_test, y_train, y_test):
    decision_tree = MyDecisionTree(X_train, X_test, y_train, y_test)
    accuracy = decision_tree.train()
    print(f'Accuracy: {accuracy}')
    print('[[TN FP]\n [FN TP]]=')
    print(decision_tree.get_confusion_matrix())
    
def gaussian_naive_bayes(X_train, X_test, y_train, y_test):
    nb = MyGaussianNaiveBayes(X_train, X_test, y_train, y_test)
    accuracy = nb.train()
    print(f'Accuracy: {accuracy}')
    print('[[TN FP]\n [FN TP]]=')
    print(nb.get_confusion_matrix())

if __name__ == '__main__':
    # patch_sklearn()
    X_train, X_test, y_train, y_test = load_data('db/Cancer_Data.csv')
    
    print('----------Logistic-----------')
    logistic_regression(X_train, X_test, y_train, y_test)
    print('----------End Logistic-----------')
    print('----------SVM-----------')
    svm(X_train, X_test, y_train, y_test)
    print('----------End SVM-----------')
    print('----------Decision Tree-----------')
    decision_tree(X_train, X_test, y_train, y_test)
    print('----------End Decision Tree-----------')
    print('----------Gaussian Naive Bayes-----------')
    gaussian_naive_bayes(X_train, X_test, y_train, y_test)
    print('----------End Gaussian Naive Bayes-----------')
