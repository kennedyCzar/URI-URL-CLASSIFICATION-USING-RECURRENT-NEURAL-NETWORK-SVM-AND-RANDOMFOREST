# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:05:01 2018

@author: kennedy
"""
__author__ = "kennedy Czar"
__email__ = "kennedyczar@outlook.com"
__version__ = '1.0'

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

"""Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter."""
    

class Classify(object):
    def __init__(self, X, Y):
        
        '''Arguments:
            X: Transformed features or predictors
            Y: Target
            '''
        self.X = X
        self.Y = np.asarray(Y)
        self.final_accuracy = []
        self.NFOLDS = 10
        self.kf = KFold(n_splits = self.NFOLDS)
        
    def Support_Vector(self):
        for self.FOLD_NO, (train_index, test_index) in enumerate(self.kf.split(self.X)):
            print('Fold: {}'.format(self.FOLD_NO))
            print("Train:", train_index, "Validation:",test_index)
            '''Split dataset into N-fold'''
            X_train = self.X[train_index]
            X_test = self.X[test_index] 
            y_train = self.Y[train_index]
            y_test = self.Y[test_index]
            '''Instantiantiate teh Naive Bayes class
            Fit data, then predict
            **We chose the category if it resulted on positive to 
            that category and negative to all others
            **If there is more than one positive or no positive, 
            then a category is chosen by random
            
            Return Classification report
                    confusion_matrix
                    precision
                    Rrecall
                    Fscore
                    Support
            '''
            clf = clf = SVC(kernel='linear')
            clf.fit(X_train, y_train)
            start_time = time()
            accuracy = clf.score(X_train, y_train)
            self.final_accuracy.append(accuracy)
            print ("training time:", round(time()-start_time, 2), "secs")
            start_time2 = time()
            prediction = clf.predict(X_test)
            '''prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')'''
            print ("predict time:", round(time()-start_time2, 2), "secs")
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
        print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(self.final_accuracy), np.std(self.final_accuracy)))
        print('*'*50)
        print('====================== 100% Support Vector Machine 100% Completed ===========================\n\n')
        
    def RandForest(self):
        print('====================== Start MultinomialNB ===========================')
        '''Random Forest Classification'''
        for self.FOLD_NO, (train_index, test_index) in enumerate(self.kf.split(self.X)):
            print('Fold: {}'.format(self.FOLD_NO))
            print("Train:", train_index, "Validation:",test_index)
            '''Split dataset into N-fold'''
            X_train = self.X[train_index]
            X_test = self.X[test_index] 
            y_train = self.Y[train_index]
            y_test = self.Y[test_index]
            '''Instantiantiate teh Multinomial Naive Bayes class
            Fit data, then predict
            **We chose the category if it resulted on positive to 
            that category and negative to all others
            **If there is more than one positive or no positive, 
            then a category is chosen by random
            
            Return Classification report
                    confusion_matrix
                    precision
                    Rrecall
                    Fscore
                    Support
            '''
            clf = RandomForestClassifier(n_estimators = 10,
                                         max_depth = 2)
            clf.fit(X_train, y_train)
            start_time = time()
            accuracy = clf.score(X_train, y_train)
            self.final_accuracy.append(accuracy)
            print ("training time:", round(time()-start_time, 2), "secs")
            start_time2 = time()
            prediction = clf.predict(X_test)
            print ("predict time:", round(time()-start_time2, 2), "secs")
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
        print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(self.final_accuracy), np.std(self.final_accuracy)))
        print('*'*50)
        print('====================== 100% Random Forest 100% Completed ===========================')
        
        
        
        
        
        
        
        
        
        
