import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from acquire import git_data
import os
import matplotlib.pyplot as plt
from string import ascii_lowercase
from itertools import product
from textblob import TextBlob
from nltk.corpus import words as english
from sklearn.model_selection import train_test_split
from prepare import git_df
from formating import bold, display, Latex, Markdown, percent, Percent, strike, underline
# Spell Checking and Visualization
from textblob import TextBlob
from wordcloud import WordCloud
import PIL
import os


###--------------- split_data ------------###


def split_data(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    return train, validate, test

###--------------- code_language ------------###


class code_language:
    def __init__(self, words, label:str):
        self.words = words
        self.label = label
        # self.freq = freq
        
    def freq(self):
        return pd.Series(self.words.split()).value_counts()
        
    def bigrams(self):
        return pd.Series(list(nltk.bigrams(self.words.split())))
    
    def trigrams(self):
        return pd.Series(list(nltk.ngrams(self.words.split(), 3)))
    
    def make_language_bank(train):
        '''
        We put train heare as the Parameter dataframe to remind the user 
        this should be used on train for exploration, after our
        train, validate, test split
        
        Returned is a Library of code_language functions for 
        bigrams, trigrams, and examining heteroskedacity 
        for the data of each programing language
        
        example---
        
        lb = make_language_bank(train)
        
        lb['Python'].label -> 'Python'
        lb['Python'].words -> '---All the words from Python Readme's---'
        
        lg.bigrams()       -> a pd.Series with a list of bigrams
        '''
        # for each language we will join all the readme entrys as words, and with their labels
        python = code_language(words= ' '.join(train[train.language == 'Python'].true_clean),
                               label= 'Python')
        html = code_language(words= ' '.join(train[train.language == 'HTML'].true_clean),
                             label= 'HTML')
        c = code_language(words= ' '.join(train[train.language == 'C'].true_clean),
                          label= 'C')
        cplusplus = code_language(words= ' '.join(train[train.language == 'C++'].true_clean),
                                  label= 'C++')
        php = code_language(words= ' '.join(train[train.language == 'PHP'].true_clean),
                            label= 'PHP')
        other = code_language(words= ' '.join(train[train.language == 'Other'].true_clean),
                              label= 'Other')
        all_langs = code_language(words= ' '.join(train.true_clean),
                                 label= 'All')

        language_bank = {'Python': python,
                         'HTML': html,
                         'C': c,
                         'C++': cplusplus,
                         'PHP': php,
                         'Other': other,
                         'All': all_langs}
        return language_bank


###--------------- other_languages ------------###


def other_languages(df):
    '''
    Takes in the DataFrame with the languages all defined and 
    spits it out with PHP, C++ Python, C, HTML, and Other
    '''
#     as above so below
#       we use map to map the langauges we want and fillna with Other
    df.language = df.language.map({'PHP':'PHP', 
                 'C++':'C++',
                 'Python':'Python',
                 'C':'C', 
                 'HTML':'HTML'}).fillna('Other')
#     return the modified DataFrame

    return df

###--------------- word_counts ------------###

def word_counts(lb):
    '''
    Parameter :
    ------------
    lb is the language_bank made by code_language
    
    returns a dataframe of the word counts of every category of languages, including all
    
    [python, html, c, c++, php, other, all]
    
    ... but it should work if you change the languages in the language bank too
    
    '''
    
    word_count = pd.DataFrame()
    columns = []
    for i in lb:
        word_count = pd.concat([word_count, lb[i].freq()], axis=1).fillna(0).astype(int)
        columns.append(i.lower())
    
    word_count.columns = columns
    
    return word_count

###------------- chances  ------------###

def chances(language_value_counts):
    '''
    
    '''
    best_chance = .166666
    
    for i, j in enumerate(language_value_counts):
        print(f' For {bold(language_value_counts.index[i])}, \
there\'s a {bold(str(percent(j/language_value_counts.sum())))} chance of that being correct')
        if j/language_value_counts.sum() > best_chance:
            best_guess = language_value_counts.index[i]
            best_chace = j/language_value_counts.sum()
        
    print(f'\n\nBest Guess:  {bold(underline(str("C++")))}, \
which gives us a {bold(underline(str(Percent(.1806))))} chance of being correct.')
    
    
def top_20_percentages(lb):
    all_word_counts = word_counts(lb)
    
    (all_word_counts.sort_values('all', ascending=False)
     .head(20)
     .apply(lambda row: row/row['all'], axis = 1)
     .drop(columns = 'all')
     .sort_values(by = 'python')
     .plot.barh(stacked = True, width = 1, ec = 'k')
    )
    plt.title('Percentage of All Top 20 words in Each Programming language\
    \nPython seems to be in Readme\'s Everywhere!', fontsize= 24)
    plt.show()