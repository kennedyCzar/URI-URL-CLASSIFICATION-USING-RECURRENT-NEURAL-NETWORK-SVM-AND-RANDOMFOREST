# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:30:06 2018

@author: kennedy

Wiley Online Library
Help paper:
    https://onlinelibrary.wiley.com/doi/full/10.1111/coin.12158
    
    
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@outlook.com"
__version__ = '1.0'

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


"""
This script Labels the data and extracts features and labels for further processing
The features are extrcted to a list and saved as feature.pkl and labels are saved as label.pkl
"""

class FETCH_DATA(object):
    def __init__(self, path, dataset):
        '''Arhuments:
            path:
                define the path containing the dataset
            dataset:
                the URI datasets we want to work on
                
        '''
        self.path = path
        self.dataset = dataset
        
    def fetch(self):
        if os.path.exists(self.path):
            try:
                print('=============Preprocessing the data=======================')
    #            path = 'D:\\FREELANCER\\CATEGORICAL_URI\\DATASET'
                
    #            os.path.exists('D:\\FREELANCER\\CATEGORICAL_URI\\DATASET')
    #            path = 'D:\\FREELANCER\\CATEGORICAL_URI\\DATASET'
#                dataset = '\\2013_04_21.csv'
                
                data = pd.read_csv(self.path + self.dataset)
        
                data.columns = ['index', 'URI', 'Section']
                data = data.drop(['index'], axis = 1)
                
                print('Done loading data')
                print(20*'*')
                print('Start labelling data....')
                '''Labelling the dataset'''
                lab = set(data['Section'].values)
                lab = dict(enumerate(lab,1))
                lab = dict (zip(lab.values(),lab.keys()))
                
                '''convert keys to values and values to keys.
                This helps to turn the label into numerics.
                for classification'''
                label = list(map(lab.get, list(data['Section'].values)))
                
                data['label'] = pd.Series(label).values
                data = data.loc[:, ['URI','label']]
                
                print('Done labelling data')
                print(20*'*')
                
                return data, label
            except:
                pass
            finally:
                print('finnished..part 1')


#%% FEATURE_EXTRACTION
                
class FEATURE_EXTRACTION(object):
    def __init__(self, data):
        '''
        Argument:
            data: processed returned by the fetch_data class and
            extract the features as predictors
        '''
        self.data = data
        
    def extract(self):
        print('Parsing and cleaning URI ')
        #Parsing and cleaning URI 
        self.features = []
        feature_text = list(self.data['URI'].values)
        
        for t in feature_text:
            if type(t) != str:
                 t = t.decode("UTF-8").encode('ascii','ignore')
                 
            t = re.sub(r'[^a-zA-Z]',r' ',t)
            
            '''you may want to include as many suffix as possible.
            This would have a positive effect on our prediction accuracy'''
            '''Another thing to do is to scrap all url suffix from Wiki'''
            
            del_words = ['www','http','com','co','uk','org',
                         'https', 'html', 'ca', 'ee', 'htm',
                         'net', 'edu', 'index', 'asp', 'au', 'nz',
                         'txt', 'php', 'de', 'cgi', 'jp', 'hub',
                         'us', 'fr', 'webs']
            
            stop_words = set(stopwords.words("english"))
            stop_words.update(del_words)
            
            '''strip the words. and remove the stopwords in our URI strings'''
            text = (i.strip() for i in t.split())
            text = [t for t in text if t not in stop_words]
            ''''join the words together'''
            text = " ".join(text)
            
            '''Append the result to the empty feature list. This would serve as our feauture
            for training and testing'''
            self.features.append(text)
            
        print('Done')
        print(20*'*')
        print('============== 100% COMPLETE ============')
        
        return self.features
    
    
    
        
    '''or you could split the dataset into smaller fraction
    by doing the following'''
    
    '''split the dataset into small fraction for test purpose..
    
    feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size = 0.99)
    
    
    the feature test is now what we make use of as our main
    feature vectore
    
    ssame goes to label_test.
    
    with open('feature_train.pkl', 'wb') as f:
        pickle.dump(feature_train, f)
    
    with open('label_train.pkl', 'wb') as f:
        pickle.dump(label_train, f)
        
    '''
    
    '''
    Save the different files. for Export only.
    
    feature_dataframe = pd.DataFrame(features, columns = ['URI'])
    label_dataframe = pd.DataFrame(label, columns = ['label'])
    #label_dataframe = label_dataframe.as_matrix(columns=None)
    processed_dataset = pd.concat([feature_dataframe, label_dataframe], axis = 1)
    processed_dataset.to_csv(path+'\\processed_URI.csv')
    
    X_train, X_test, Y_train, Y_test = train_test_split(feature_dataframe, label_dataframe, test_size = 0.2)
    X_train.to_csv(path+'\\X_train.csv')
    X_test.to_csv(path+'\\X_test.csv')
    Y_train.to_csv(path+'\\Y_train.csv')
    Y_test.to_csv(path+'\\Y_test.csv')
    
    '''
    



#%% CONVERT CATEGORICAL DATA TO NUMERICS

class PREPROCESS(object):
    def __init__(self, X, Y):
        '''Arguments:
            X: feacture vector
                Y: scalar to be predicted
                    '''
        self.X = X
        self.Y = Y
        
    def process(self):
        '''Initialize TfidfVectorizer:
            TfidfVectorizer: Converst our categorical feature into numerical vectors
            '''
        vectorizer = TfidfVectorizer()
        features_train_transformed = vectorizer.fit_transform(self.X)
        
        print('Vectorizing completes....')
        print('Performing SelectPercentile completes....')
        '''SelectPercentile provides an automatic procedure for 
        keeping only a certain percentage of the best, associated features.
        f_classif: Used only for categorical targets and based on the 
        Analysis of Variance (ANOVA) statistical test.
        '''
        selector = SelectPercentile(f_classif, percentile=25)
        selector.fit(features_train_transformed, self.Y)
        features_train_ = selector.transform(features_train_transformed).toarray()
        
        print('SelectPercentile completes....')
        return features_train_





















