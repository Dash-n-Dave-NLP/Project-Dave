# NLP Team Project: Predicting a Github Repository's Programming Language Using NLP

A recording of our presentation can be viewed here: https://www.canva.com/design/DAFPgY6Aj6s/NMmhxDsXu_SjriGNlGZ3-w/view?utm_content=DAFPgY6Aj6s&utm_campaign=designshare&utm_medium=link&utm_source=recording_view


## Goals

The purpose of this project is to analyze the readme files of github repositories and develop a machine learning model to predict the programming language used in the repository using natural language processing. We obtained the dataset for this project from https://www.github.com. The code for acquiring this dataset is in our acquire.py file.

We are using this dataset for academic purposes only.

Initial Questions:

- What keywords in a readme readily identify a particular programming language?
- Can certain word combinations / n-grams in a readme improve identification of programming languages? 

## Executive Summary

- We acquired 680 readme files from repositories on Github and removed duplicates and nulls. The readme contents were cleaned, tokenized, stemmed, lemmatized, and stopwords were removed. The final cleaned and prepared dataset was 600 observations. We decided to use five major programming languages: Python, C, C++, HTML, and PHP. All other programming languages were grouped into a category called "other." 
- The dataset was split into train, validate, and test using a 60/20/20 split stratified on language. 
- We trained and evaluated three models: Decision Tree Classifier, Logistic Regression, and Naive-Bayes Multinomial Classifier. For each model we used a Count Vectorizer, Count Vectorizer with bigrams, and a TF-IDF Vectorizer. The TF-IDF Vectorizer produced the best overall accuracy in each model.
- The selected model is a Logistic Regression model using a TF-IDF Vectorizer. The model performed at 96 percent accuracy on train, but accuracy dipped to 54 percent on the validate set. When model performed at 58 percent accuracy on the test set. This is 41 percent above the baseline accuracy, which is 17 percent.

## Data Dictionary

1. repo : the namepath of the repository (string)
2. language : programming language of the repository (string)
3. original : readme file (string)
4. clean : basic clean version of original (string)
5. stemmed : cleaned and stemmed version of original (string)
6. lemmatized: cleaned and lemmatized version of original (string)
7. original_length: number of words in each original observation (int64)
8. true_clean : cleaned and lemmatized version of original with stopwords and non-dictionary words removed (string)


## Project Planning

- Acquire the data from Github and save to a local pickle file
- Prepare the data with the intent to discover the main predictors of programming language; clean the data and engineer features if necessary; ensure that the data is tidy
- Split the data into train, validate, and test datasets using a 60/20/20 split and a random seed of 217
- Explore the data:
    - Find top 20 words for each programming language
    - Find top 20 bigrams for each programming language
- Create graphical representations of the analyses
- Ask more questions about the data
- Document findings
- Train and test models:
    - Baseline accuracy with "other" language category is 17 percent; with "other" removed, baseline accuracy is 21 percent
    - Select vectorizer and train multiple classification models
    - Test the model on the validate set, adjust model parameters if necessary
- Select the best model for the project goals:
    - Determine which model performs best on the validate set
- Test and evaluate the model:
    - Use the model on the test set and evaluate its performance (accuracy, precision, recall, f1, etc.)
- Visualize the model's performance on the test set
- Document key findings and takeaways, answer the questions
- Create a final report

## How to Reproduce this Project

- In order to reproduce this project, you will need access to Github or the data file. Acquire the database from https://www.github.com using the function in our acquire.py file. The prepare.py file has the necessary functions to prepare and split the dataset.

- You will need to import the following python libraries into a python file or jupyter notebook:

    - import pandas as pd
    - import numpy as np
    - import matplotlib.pyplot as plt
    - import seaborn as sns
    - from scipy import stats
    - from sklearn.model_selection import train_test_split
    - from sklearn.tree import DecisionTreeClassifier, plot_tree
    - from sklearn.metrics import classification_report, accuracy_score
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    - from sklearn.naive_bayes import MultinomialNB, CategoricalNB
    - import re
    - import unicodedata
    - import nltk
    - from wordcloud import WordCloud
    - from requests import get
    - from bs4 import BeautifulSoup
    - from nltk.corpus import stopwords
    - import acquire
    - import prepare
    - import model
    - import warnings
    - warnings.filterwarnings("ignore")

- Prepare and split the dataset. The code for these steps can be found in the acquire.py file and prepare.py file within this repository.

- Use pandas to explore the dataframe and matplotlib to visualize word counts and n-grams.

- Use WordCloud to create wordcloud images for each programming language.

- Analyze words in each programming language to find the most used words.

- Create models (decision tree, Naive-Bayes classifier, and logistic regression) using sklearn.

- Train each model and evaluate its accuracy on both the train and validate sets.

- Select the best performing model and use it on the test set.

- Graph the results of the test using probabilities.

- Document each step of the process and your findings.


## Key Findings and Takeaways

After training and evaluating three models using both a single-word count vectorizer, bigram count vectorizer, single-word TF-IDF vectorizer, and bigram TF-IDF vectorizer, the balanced logistic regression model provdided the best overall performance on the validate set both when the "other" language category is removed and when the category is retained. We decided to retain the "other" category to account for languages outside the five major categories. Fitting of the models resulted in over 90 percent accuracy on train; however, the accuracy of all models fell considerably on the validate set. The selected model performs with a 41 percent accuracy over baseline. Bigrams did not improve the model as much as we anticipated, and some models performed worse with both the bigram CV and TF-IDF. The selected model was trained using the TF-IDF vectorizer with single words only.

We set out to see if we could increase our ability to predict the Programing Language of a github Repo by looking at their Readme files.
We established a Baseline of 18% Acc. and increased to 58% Acc. on out of training data.

Next Steps

We want to do more with our Data Engineering:
    - Trying to isolate the most common words from everything and get rid of them
    - Making sure we are not eliminating words or urls that may be Programing language specific
    - Procure a larger sample that has a proper representation of the Population
Additional Lifting Power:
    - We want to use the API Key we aquired for GPT-3 to summarize the Readme information
    - If the different languages have similar themes, the transformed information could give us more information


