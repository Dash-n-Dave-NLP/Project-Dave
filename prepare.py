import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from acquire import git_data
import os
from string import ascii_lowercase
from itertools import product
from textblob import TextBlob
from nltk.corpus import words as english
from sklearn.model_selection import train_test_split



def get_letters ():
    letters = []

    for length in range(1, 3):
        for combo in product(ascii_lowercase, repeat=length):
            letters.append(''.join(combo))
            
    return letters


def basic_clean(string):
    clean = string.lower()
    clean = unicodedata.normalize('NFKD', clean)\
    .encode('ascii', 'ignore')\
    .decode('utf-8')
    clean = re.sub(r'[^a-z\'\s]', '', clean)
    
    return clean

def tokenize(string):
    toke = nltk.tokenize.ToktokTokenizer()
    toked = toke.tokenize(string, return_str=True)
    return toked

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stemmed = [ps.stem(word) for word in string.split()]
    stemmed = ' '.join(stemmed)
    return stemmed

def lemmatize(string):
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemman = [wnl.lemmatize(word) for word in string.split()]
    
    output_lemman = ' '.join(lemman)
    
    return output_lemman

def remove_stopwords(string):
    stopper = stopwords.words('english')
    stopper.append("'")
    stopper.extend(['http','https','img','png', 'github'])
    stopper.extend(get_letters())
    my_words = string.split()
    dont_stop = [word for word in my_words if word not in stopper]
    unstopped = ' '.join(dont_stop)
    return unstopped

# def spell_check(string):
#     checker = TextBlob(string)
#     checked = str(checker.correct())
#     return checked

def spell_check(string):
    vocab = set(w.lower() for w in english.words())
    checker = string.split()
    checked = [word for word in checker if word in vocab]
    checked_string = ' '.join(checked)
    return checked_string

def make_original(dataframe):
    for i in dataframe:
        if i == 'readme_contents':
            dataframe['original'] = dataframe['readme_contents'].copy()
            dataframe.drop(columns= 'readme_contents', inplace=True)
        if i == 'body':
            dataframe['original'] = dataframe['body'].copy()
            dataframe.drop(columns= 'body', inplace=True)
        if i == 'content':
            dataframe['original'] = dataframe['content'].copy()
            dataframe.drop(columns= 'content', inplace=True)
    return dataframe

def make_clean(dataframe):
    dataframe['clean'] = dataframe['original'].apply(basic_clean).apply(tokenize).apply(remove_stopwords)
    return dataframe

def make_stemmed(dataframe):
    dataframe['stemmed'] = dataframe['clean'].apply(stem)
    return dataframe

def make_lemmatized(dataframe):
    dataframe['lemmatized'] = dataframe['clean'].apply(lemmatize)
    return dataframe

def make_nlp_df():
    '''
    No Parameters:
    --------------
    
    Returns 2 Dataframes:   codeup_df, news_df
    
    Both will contain columns ['title', 'content', 'original', 'clean', 'stemmed', 'lemmatized']
    '''
    codeup_df = get_blog_articles()
    news_df = get_all_shorts()
    
    codeup_df = make_lemmatized(make_stemmed(make_clean(make_original(codeup_df))))
    news_df = make_lemmatized(make_stemmed(make_clean(make_original(news_df))))
    
    return codeup_df, news_df


def git_df():
    '''
    Parameters :
    --------------
    df         :   DataFrame with the site
    '''
    # Set the filename for caching
    filename= 'clean_df'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        df = df.drop(columns='index')        
        
    else:
        df = git_data(df=True)

        df = make_lemmatized(make_stemmed(make_clean(make_original(df))))

        df['original_length'] = df.original.apply(len)

        df = df.reset_index()

        df['true_clean'] = pd.Series([' '.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in df.original]).apply(basic_clean).apply(str.strip).apply(tokenize).apply(remove_stopwords).apply(lemmatize)
        
        df.drop(index= 105, inplace=True)
        
        df = df.reset_index()
        
        df['true_clean'] = df['true_clean'].str.findall('\w{,12}').str.join(' ').str.findall('\w{4,}').str.join(' ')
        
        df['true_clean'] = df['true_clean'].apply(spell_check)

        pd.to_pickle(df, 'clean_df')
    return df

def split_data(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    return train, validate, test


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

