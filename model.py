import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
import warnings
warnings.filterwarnings("ignore")

def tree_model(X_train, y_train, X_validate, y_validate, depth):
    '''This function takes in five arguments: a dataframe for the training features, a dataframe for the training target, 
        a dataframe for the validation features, a dataframe for the validation target, and the depth as an integer. The
        function fits a DecisionTreeClassifier to train with the specified maximum depth and plots the tree. Then it 
        predicts values from the training set and calculates the model's performance. It returns a statement of the model's
        accuracy on train and validate and prints a classification report for the train performance.'''
    # initiate the decision tree model and assign to a variable
    clf = DecisionTreeClassifier(max_depth=depth, random_state=217)
    # fit the model to train
    clf = clf.fit(X_train, y_train)
    # designate the size of the tree visualization
    plt.figure(figsize=(13, 7))
    # show a graphical representation of the decision tree
    plot_tree(clf)
    plt.show()
    # get predictions for the model and save to a variable
    y_pred = clf.predict(X_train)
    # get probabilities for the model and save to a variable
    y_pred_proba = clf.predict_proba(X_train)
    # print the accuracy of the model on train
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    # print the classification report for the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate to check for overfitting
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'.format(clf.score(X_validate, y_validate)))

def rand_forest(X_train, y_train, X_validate, y_validate, depth, samples):
    '''This function takes in six arguments: training dataframe of features, training dataframe for the target, validation
        dataframe of features, validation dataframe for target, the depth of the tree as an integer, and the minimum number 
        of samples per leaf as an integer. The function fits and trains a random forest model and makes predictions for 
        testing. The function returns a classification report of the model's performance on the training set as well as the 
        depth and sample leaf size. It also prints the model's accuracy score on train and validate.'''
    # Make the model
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=samples, random_state=217)
    # Fit the model on train only
    forest.fit(X_train, y_train)
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred = forest.predict(X_train)
    # get probabilities for the model and save to a variable
    y_pred_proba = forest.predict_proba(X_train)
    # print the accuracy of the model on train
    print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(forest.score(X_train, y_train)))
    # print the classification report for the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate to check for overfitting
    print('Accuracy of Random Forest classifier on validate set: {:.2f}'.format(forest.score(X_validate, y_validate)))

def bayes_model(X_train, y_train, X_validate, y_validate):
    '''This function takes in four arguments: training features dataframe, training target dataframe, validation features
        dataframe, and validation target dataframe. The function makes a Naive-Bayes multinomial classifier and fits in to train. 
        The model makes predictions on the training set and evaluates its performance. The function returns the accuracy of
        the model on the train and validate sets, as well as a classification report of the model's performance on train.'''
    # Create the logistic regression
    bayes = MultinomialNB(alpha=.5, fit_prior=True)
    # Fit a model using only the specified features
    bayes.fit(X_train, y_train)
    # Predict on that same subset of features
    y_pred = bayes.predict(X_train)
    # print a title
    print("Multinomial Naive-Bayes")
    # print the accuracy of the model on train
    print('Accuracy of Multinomial Naive-Bayes classifier on training set: {:.2f}'.format(bayes.score(X_train, y_train)))
    # print a classification report of the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'.format(bayes.score(X_validate, y_validate)))

def log_model_balanced(X_train, y_train, X_validate, y_validate):
    '''This function takes in four arguments: training features dataframe, training target dataframe, validation features
        dataframe, and validation target dataframe. The function makes a balanced logistic regression model and fits in to train. 
        The model makes predictions on the training set and evaluates its performance. The function returns the accuracy of
        the model on the train and validate sets, as well as a classification report of the model's performance on train.'''
    # Create the logistic regression with balanced class weight
    logit = LogisticRegression(random_state=217, class_weight='balanced')
    # Fit a model using only the specified features
    logit.fit(X_train, y_train)
    # Predict on that same subset of features
    y_pred = logit.predict(X_train)
    # print a title
    print("Logistic Regression using selected features")
    # print the accuracy of the model on train
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(logit.score(X_train, y_train)))
    # print a classification report of the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'.format(logit.score(X_validate, y_validate)))

def log_model(X_train, y_train, X_validate, y_validate):
    '''This function takes in four arguments: training features dataframe, training target dataframe, validation features
        dataframe, and validation target dataframe. The function makes a logistic regression model and fits in to train. 
        The model makes predictions on the training set and evaluates its performance. The function returns the accuracy of
        the model on the train and validate sets, as well as a classification report of the model's performance on train.'''
    # Create the logistic regression
    logit = LogisticRegression(solver='saga', penalty='elasticnet', random_state = 217, 
                            class_weight='balanced', l1_ratio=.001).fit(X_train, y_train)
    # Fit a model using only the specified features
    logit.fit(X_train, y_train)
    # Predict on that same subset of features
    y_pred = logit.predict(X_train)
    # print a title
    print("Logistic Regression using selected features")
    # print the accuracy of the model on train
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(logit.score(X_train, y_train)))
    # print a classification report of the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'.format(logit.score(X_validate, y_validate)))

def test_log(X_train, y_train, X_test, y_test):
    '''This function takes in six arguments: training dataframe of features, training dataframe for the target, validation
        dataframe of features, validation dataframe for target, the depth of the tree as an integer, and the minimum number 
        of samples per leaf as an integer. The function fits and trains a logistic regression model and makes predictions for 
        testing. The function returns a classification report of the model's performance on the training set and test
        set, as well as the depth and sample leaf size. It also prints the model's accuracy score on train and test.'''
    
    # Make the model
    lm = LogisticRegression(solver='saga', penalty='elasticnet', random_state = 217, 
                            class_weight='balanced', l1_ratio=.001).fit(X_train, y_train)
    # Fit the model on train only
    lm.fit(X_train, y_train)
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred = lm.predict(X_train)
    # get probabilities for the model and save to a variable
    y_pred_proba = lm.predict_proba(X_train)
    # evaluate the model's performance on the test set
    y_pred_test = lm.predict(X_test)
    # print the accuracy of the model on train
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(lm.score(X_train, y_train)))
    # print the classification report for the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate to check for overfitting
    print('Accuracy of Logistic Regression classifier on test set:     {:.2f}'.format(lm.score(X_test, y_test)))
    print(classification_report(y_test, y_pred_test))

def test_log_balanced(X_train, y_train, X_test, y_test):
    '''This function takes in six arguments: training dataframe of features, training dataframe for the target, validation
        dataframe of features, validation dataframe for target, the depth of the tree as an integer, and the minimum number 
        of samples per leaf as an integer. The function fits and trains a logistic regression model and makes predictions for 
        testing. The function returns a classification report of the model's performance on the training set and test
        set, as well as the depth and sample leaf size. It also prints the model's accuracy score on train and test.'''
    
    # Make the model
    lm = LogisticRegression(random_state = 217, class_weight='balanced').fit(X_train, y_train)
    # Fit the model on train only
    lm.fit(X_train, y_train)
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_pred = lm.predict(X_train)
    # get probabilities for the model and save to a variable
    y_pred_proba = lm.predict_proba(X_train)
    # evaluate the model's performance on the test set
    y_pred_test = lm.predict(X_test)
    # print the accuracy of the model on train
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'.format(lm.score(X_train, y_train)))
    # print the classification report for the model's performance on train
    print(classification_report(y_train, y_pred))
    # print the accuracy of the model on validate to check for overfitting
    print('Accuracy of Logistic Regression classifier on test set:     {:.2f}'.format(lm.score(X_test, y_test)))
    print(classification_report(y_test, y_pred_test))    

