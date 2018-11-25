# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:49:39 2018

@author: kennedy

Help paper:
    https://onlinelibrary.wiley.com/doi/full/10.1111/coin.12158
    
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@outlook.com"
__version__ = '1.0'

import pandas as pd
import numpy as np
from os import chdir
import os
from sklearn.model_selection import train_test_split, KFold

os.path.exists('D:\\FREELANCER\\CATEGORICAL_URI\\DATASET')
path = 'D:\\FREELANCER\\CATEGORICAL_URI\\DATASET'
data = '\\2013_04_21.csv'

dataset = pd.read_csv(path + data)
dataset.columns = ['index', 'URI', 'Section']
dataset = dataset.drop(['index'], axis = 1)
X_train, X_test = train_test_split(dataset, test_size = 0.002)


'''Work with X_test'''
new_dataset = X_test

new_dataset.URI.iloc[5]
'''strip the elemenets in the URI'''
stripped = new_dataset.URI.iloc[0:].strip('http://www.')
striptail = stripped.strip('.com/')
len(new_dataset.URI)



import re
def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)
   
text = re.sub(r'^https?:\/\/.*[\r\n]*', '', new_dataset.URI.iloc[0], flags=re.MULTILINE)
#rev = remove_urls(new_dataset.URI.iloc[0])

#%% Using the Library: tldextract


import tldextract

ext = tldextract.extract(new_dataset.URI.iloc[5])
tldextract.extract(new_dataset.URI.iloc[0])


from urllib.parse import urlparse
url = urlparse(new_dataset.URI.iloc[5])

split_netloc = url.netloc.split('.')
split_path = url.path.split('/')



unwanted_urls = ['org', 'com', 'ee', 'edu', 'net', 'au', 'html',
                 'tw', 'co', 'uk', 'htm', '']



for ii in unwanted_urls:
    for ij in split_netloc:
        if ii == ij:
            split_netloc.remove(ii)
    
    
    
    
import bs4 as bs
import pickle
import requests
def extract_domain():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': ['wikitable sortable', 'wikitable']})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("domain.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

domain = extract_domain()


        
unwanted_urls = ['org', 'com', 'ee', 'edu', 'net', 'au', 'html',
                 'tw', 'co', 'uk', 'htm', '']

for ii in new_dataset.URI.head(5):
    print(ii)
    complete = []
    url = urlparse(ii)
    split_netloc = url.netloc.split('.')
    split_path = url.path.split('/')
    for ij in split_netloc:
        complete.append(ij)
        
    for ik in split_path:
        
        complete.append(ik)
    


#%% USING THE NLTK LIBRARY
        
import pandas as pd
import pickle
from nltk.corpus import stopwords
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
"""
This script Labels the data and extracts features and labels for further processing
The features are extrcted to a list and saved as feature.pkl and labels are saved as label.pkl
"""


os.path.exists('D:\\FREELANCER\\CATEGORICAL_URI\\DATASET')
path = 'D:\\FREELANCER\\CATEGORICAL_URI\\DATASET'
dataset = '\\2013_04_21.csv'

data = pd.read_csv(path + dataset)
#data = data.loc[:,['Primary Category', 'URL']]

data.columns = ['index', 'URI', 'Section']
data = data.drop(['index'], axis = 1)

#Labelling the data
lab = set(data['Section'].values)
lab = dict(enumerate(lab,1))
lab = dict (zip(lab.values(),lab.keys()))

label = list(map(lab.get, list(data['Section'].values)))

data['label'] = pd.Series(label).values
data = data.loc[:, ['URI','label']]

with open('data.pkl','wb') as f:
    pickle.dump(data, f)
    
with open('label.pkl', 'wb') as f:
    pickle.dump(label, f)


#Parsing and cleaning URL(Features)    
feature_text = list(data['URI'].values)

features = []
for t in feature_text:
    if type(t) != str:
         t = t.decode("UTF-8").encode('ascii','ignore')
         
    t = re.sub(r'[^a-zA-Z]',r' ',t)

    del_words = ['www','http','com','co','uk','org','https']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    text = (i.strip() for i in t.split())
    text = [t for t in text if t not in stop_words]
    text = " ".join(text)
    
    features.append(text)


'''split the dataset into small fraction for test purpose..'''
feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size = 0.002)


with open('feature_test.pkl', 'wb') as f:
    pickle.dump(feature, f)

with open('label_test.pkl', 'wb') as f:
    pickle.dump(label, f)
    
    
#label_train, label_test = train_test_split(label, test_size = 0.2)
#%% SAVE FEATURES AND LABELS
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




from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport

 """Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter."""
    
clf = GaussianNB()
clf.fit(features_train_transformed, labels_train)

t0 = time()
accuracy = clf.score(features_test_transformed, labels_test)
print ("training time:", round(time()-t0, 3), "s")


visualizer = ClassificationReport(clf, classes=target_names, support=True)
visualizer.fit(features_train_transformed, labels_train)
accuracy = visualizer.score(features_test_transformed, labels_test)
visualizer.poof()


t0 = time()
prediction = clf.predict(features_test_transformed)
print ("predict time:", round(time()-t0, 3), "s")

#print(confusion_matrix(labels_test, prediction, labels=['Recreation', 'Shopping',
#                                                        'Reference', 'Sports', 'Computers', 
#                                                        'News', 'Games', 'Home', 'Arts',
#                                                        'Society', 'Science', 'Business',
#                                                        'Health']))

target_names = ['Recreation', 'Shopping',
        'Reference', 'Sports', 'Computers', 
        'News', 'Games', 'Home', 'Arts',
        'Society', 'Science', 'Business',
        'Health']
print(classification_report(labels_test, prediction, target_names = target_names))
print(confusion_matrix(labels_test, prediction, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
print(precision_recall_fscore_support(labels_test, prediction))



#%% CROSS VALIDATION
N_SPLITS = 8 
kf = KFold(n_splits = N_SPLITS)
count = 0
for train_index, test_index in kf.split(features_train_transformed):
    print("Train:", train_index, "Validation:",test_index)
    X_train, X_test = features_train_transformed[train_index], features_train_transformed[test_index] 
    y_train, y_test = labels_train[train_index], labels_train[test_index]
    while count <= N_SPLITS:
        print('Fold: {}'.format(count))
        clf = GaussianNB()
        clf.fit(features_train_transformed, labels_train)
        t0 = time()
        accuracy = clf.score(features_test_transformed, labels_test)
        print ("training time:", round(time()-t0, 3), "s")
        t0 = time()
        prediction = clf.predict(features_test_transformed)
        print ("predict time:", round(time()-t0, 3), "s")
        target_names = ['Recreation', 'Shopping',
                'Reference', 'Sports', 'Computers', 
                'News', 'Games', 'Home', 'Arts',
                'Society', 'Science', 'Business',
                'Health']
        print(classification_report(labels_test, prediction, target_names = target_names))
        print(confusion_matrix(labels_test, prediction, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
        print(precision_recall_fscore_support(labels_test, prediction))
        count += 1
 

for k in range(2,10):
    print('===FOLD {}===='.format(k))
    result = cross_val_score(clf, features_train_transformed, labels_train, cv=k, scoring='r2')
    print(k, result.mean())
    y_pred = cross_val_predict(clf, features_train_transformed, labels_train, cv=k)
    print(y_pred)
#    print(classification_report(labels_test, y_pred, target_names = target_names))
#    print(confusion_matrix(labels_test, y_pred, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
#    print(precision_recall_fscore_support(labels_test, y_pred))
    
    
from sklearn import cross_validation 
kf_n = cross_validation.KFold(features_train_transformed.shape[0], n_folds=8, shuffle=True)

yV_pred = cross_validation.cross_val_predict(clf, features_test_transformed, labels_test, cv = kf_n, n_jobs = 1)

print(classification_report(labels_test, yV_pred, target_names = target_names))
print(confusion_matrix(labels_test, prediction, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
print(precision_recall_fscore_support(labels_test, prediction))

#from sklearn.cross_validation import cross_val_score, cross_val_predict
#from sklearn.model_selection import cross_validate
#clf = GaussianNB()
for CV in range(2, 8):
    print('=====FOLD {}======'.format(CV))
    clf = GaussianNB()
    clf.fit(features_train_transformed, labels_train)
    score = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    acc_scores = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    auc_scores = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    prec_scores = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    recall_scores = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    f1_scores = cross_validate(clf, features_train_transformed, labels_train, cv = CV)
    print('acc\t {}'.format(np.mean(acc_scores)))
    print('auc\t {}'.format(np.mean(auc_scores)))
    print('prec\t {}'.format(np.mean(prec_scores)))
    print('recall\t {}'.format(np.mean(recall_scores)))
    print('f1\t {}'.format(np.mean(f1_scores)))
    t0 = time()
    accuracy = clf.score(features_test_transformed, labels_test)
    print ("training time:", round(time()-t0, 3), "s")
    t0 = time()
    prediction = cross_val_predict(features_test_transformed)
    print ("predict time:", round(time()-t0, 3), "s")
    print(sorted(score.keys()) )
    target_names = ['Recreation', 'Shopping',
            'Reference', 'Sports', 'Computers', 
            'News', 'Games', 'Home', 'Arts',
            'Society', 'Science', 'Business',
            'Health']
    print(classification_report(labels_test, prediction, target_names = target_names))
    print(confusion_matrix(labels_test, prediction, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
    print(precision_recall_fscore_support(labels_test, prediction))
   

from sklearn.metrics import accuracy_score
def _calculate(X, y):
        import sklearn.naive_bayes
        
        kf = cross_validation.KFold(features_train_transformed.shape[0], n_folds=8)

        accuracy = 0
        for train, test in kf:
            nb = sklearn.naive_bayes.GaussianNB()
            nb.fit(X[train], y[train])
            predictions = nb.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10 
    
_calculate(features_train_transformed, labels_train) 

  
'''
Rough estimate of NB Classification
Classification of 13 classes.


[[ 421 2577   10   20   29   32    4   21   19   14   17   59   38]
 [  20 2889    8    6   13    7    0    9   15    9    3   41   13]
 [  70  854  360   91  166   57    6    3   19   47   54  119  173]
 [  74 2115   22 1036   24   42    7    4   25    8   10   62   20]
 [  79 2399   31   19  875   54   17   10   15   18   53  166   64]
 [   2  244    4    2    2   31    0    1    1    1    4    3    5]
 [  62  588    5   38   77   10  777    1  125   17   10   28   22]
 [  41  381    8   11   41    9    3  315    8    8   17   15   27]
 [ 327 4519   48   72  182  189   45   12 2376  107   42  175  133]
 [ 267 4978   71   55  189  202   17   21  111 1586   68  194  255]
 [ 119 1558  141   37  309  121   16   22   48   67 1000  239  225]
 [  56 7206   23   10   48   41   15   21   15   11   29  461   54]
 [  33 1227   13   15   15   21    3    5   10   15   18   47  519]]

'''

#%%

from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport


def preprocess_and_model():
    
    with open('feature_test.pkl', 'wb') as f:
        pickle.dump(feature_test, f)
    
    with open('label_test.pkl', 'wb') as f:
        pickle.dump(label_test, f)
        
#    features_train, features_test, labels_train, labels_test = \
#    train_test_split(feature_test, label_test, test_size=0.2, random_state=42)
#    N_SPLITS = 8
    NTRAIN = 100
    NTEST = 0
    X_train = []
    Y_train = []
    Y_test = []
    X_test = []
    NFOLDS = 8
    SEED = 0
    kf = KFold(n_splits = NFOLDS)
    count = 0
    kf = KFold(NTRAIN, shuffle=True, random_state=SEED)
    test_label = np.asarray(labels_train)
    
    final_accuracy = []
    for FOLD_NO, (train_index, test_index) in enumerate(kf.split(features_train_transformed)):
        print('Fold: {}'.format(FOLD_NO))
        print("Train:", train_index, "Validation:",test_index)
        X_train = features_train_transformed[train_index]
        X_test = features_train_transformed[test_index] 
        y_train = test_label[train_index]
        y_test = test_label[test_index]
#        while count <= NFOLDS:
#        for fold_no in range(NFOLDS):
#        print('Fold: {}'.format(fold_no))
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        t0 = time()
        accuracy = clf.score(X_train, y_train)
        final_accuracy.append(accuracy)
        print ("training time:", round(time()-t0, 3), "s")
        t0 = time()
        prediction = clf.predict(X_test)
        print ("predict time:", round(time()-t0, 3), "s")
        target_names = ['Recreation', 'Shopping',
                'Reference', 'Sports', 'Computers', 
                'News', 'Games', 'Home', 'Arts',
                'Society', 'Science', 'Business',
                'Health']
        
        print('==== CLASSIFICATION REPORT ======')
        print(classification_report(y_test, prediction, target_names = target_names))
        print('*'*50)
        print('==== CONFUSION MATRIX ======')
        print(confusion_matrix(y_test, prediction, labels = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
        print('*'*50)
        print('==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======')
        '''Note that this is same as classification report
        but without Support. And it is the weighted average'''
        print(precision_recall_fscore_support(y_test, prediction, average='weighted'))
        print('*'*50)
    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(final_accuracy), np.std(final_accuracy)))
    print('*'*50)
#            count += 1
    
#    vectorizer = TfidfVectorizer()
#    features_train_transformed = vectorizer.fit_transform(features_train)
#    features_test_transformed  = vectorizer.transform(features_test)
    
    
    
    selector = SelectPercentile(f_classif, percentile=25)
#    selector.fit(features_train_transformed, labels_train)
#    features_train_transformed = selector.transform(features_train_transformed).toarray()
#    features_test_transformed  = selector.transform(features_test_transformed).toarray()
#    
#    return features_train_transformed, features_test_transformed, labels_train, labels_test
#
#features_train_transformed, features_test_transformed, labels_train, labels_test = preprocess()

#%% Recurrent Neural Network
    
#import keras libraries
#initialize recurrent neural network
from keras.models import Sequential 
#creates output layer of our RNN
from keras.layers import Dense, Activation
from keras.layers import Flatten, Convolution1D, Dropout
#Long Short Memory RNN
from keras.layers import LSTM
#import the keras GRU
from keras.layers import GRU
#Dense, Activation and Dropout
#from keras.layers.core import Dense, Activation, Dropout
#Convolutional 1D
from keras.layers.convolutional import Conv1D
#MaxPooling1D
from keras.layers.convolutional import MaxPooling1D
#imprt optimizer
from keras.optimizers import SGD





