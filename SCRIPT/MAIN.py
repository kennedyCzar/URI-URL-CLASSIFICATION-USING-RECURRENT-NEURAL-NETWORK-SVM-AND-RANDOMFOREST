# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:05:01 2018

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@outlook.com"
__version__ = '1.0'

import pickle
from sklearn.model_selection import train_test_split
from DATA_PREPROCESS_MAIN import FETCH_DATA, FEATURE_EXTRACTION
from DATA_PREPROCESS_MAIN import PREPROCESS
from PREDICTIVE_MODEL import Classify

'''main function'''

if __name__ == '__main__':
    path = 'D:\\FREELANCER\\CATEGORICAL_URI\\DATASET'
    dataset = '\\2013_04_21.csv'
    data, label = FETCH_DATA(path, dataset).fetch()
    print('Load processed data to pickle')
    '''Load proceesed data to pickle'''
    with open('data.pkl','wb') as f:
        pickle.dump(data, f)
    
    '''load labels to pickle.'''
    with open('label.pkl', 'wb') as f:
        pickle.dump(label, f)
    print('Done..')
    print(20*'*')
    
    features = FEATURE_EXTRACTION(data).extract()
    print('Load features to pickel')
    '''save the feature with pickle in current directory.
    saves us memory because of the size on memory.'''
    with open('feature.pkl', 'wb') as f:
        pickle.dump(features, f)  #NOTE that features contains all dataset
    print('Done')
    
    from os import chdir
    chdir('D:\\FREELANCER\\CATEGORICAL_URI\\SCRIPT')
    
    '''Note that this part of the code is to split the data into a fragment i can use.
    I split the data so i can return only 0.1% of the initial data as my useable dataset.
    
    Comment out the line to use the whole dataset. Note that this requires alot of
    computational power because the dataset is over 1Million.
    Also you would most likely get a MemoryError.
    
    It would fine if you have a system configuration with >24GM RAM. 
    Mine is 24GB RAM and perhaps if you have something higher than that, you
    wouldnt have to bother with MemoryErrors.
    
    To processed if all conditions are met. do the following.
    
    comment this..
    feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size = 0.99)
    
    with open('feature.pkl', 'wb') as f:
        pickle.dump(feature, f)
    
    with open('label.pkl', 'wb') as f:
        pickle.dump(label, f)
    
    
    with open('feature.pkl','rb') as f:
        feature = pickle.load(f)
        
    with open('label_train.pkl','rb') as f:
        label = pickle.load(f)
    
    features_train_transfm = PREPROCESS(feature, label).process()
    
    Return model result
    NB(features_train_transfm, label_train)
    
    '''
    feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size = 0.99)
    with open('feature_train.pkl', 'wb') as f:
        pickle.dump(feature_train, f)
    
    with open('label_train.pkl', 'wb') as f:
        pickle.dump(label_train, f)
    
    
    with open('feature_train.pkl','rb') as f:
        feature_new = pickle.load(f)
        
    with open('label_train.pkl','rb') as f:
        label_new = pickle.load(f)
        
    '''
    Feature Vectorization:
        features_train_BNB:
            Scale, Vectorize + SelectPercentile
        features_train_BNB_WTHSelector:
            Scale, Vectorize without SelectPercentile
        features_train_MNB:
            CountVectorizer + TfidfVectorizer
            
        The reason for vectorizing using this different approaches
        is to check which maps the features to the label much better.
        
        The better would give high average precision on test data.
    '''
    features_train = PREPROCESS(feature_new, label_new).process()
    
    '''Return model result'''
    '''
    Binomial Model Output:
        NB(features_train_BNB, label_train)
    '''
    Classify(features_train, label_new).Support_Vector()
    Classify(features_train, label_new).RandForest()










