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












#'''OUTPUT
#
#=============Preprocessing the data=======================
#Done loading data
#********************
#Start labelling data....
#Done labelling data
#********************
#finnished..part 1
#Load processed data to pickle
#Done..
#********************
#Parsing and cleaning URI 
#Done
#********************
#============== 100% COMPLETE ============
#Load features to pickel
#Done
#Vectorizing completes....
#Performing SelectPercentile completes....
#SelectPercentile completes....
#Fold: 0
#Train: [ 1518  1519  1520 ... 12141 12142 12143] Validation: [   0    1    2 ... 1515 1516 1517]
#training time: 9.29 secs
#predict time: 1.37 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.60      0.25      0.35       128
#   Shopping       0.19      0.11      0.14       102
#  Reference       0.62      0.29      0.40        55
#     Sports       0.57      0.24      0.34       145
#  Computers       0.62      0.15      0.25        65
#       News       0.56      0.12      0.20        75
#      Games       0.07      0.10      0.08        10
#       Home       0.00      0.00      0.00       103
#       Arts       0.46      0.27      0.34        85
#    Society       0.22      0.95      0.35       244
#    Science       0.54      0.27      0.36        26
#   Business       0.67      0.17      0.27       233
#     Health       0.70      0.21      0.33       247
#
#avg / total       0.48      0.31      0.28      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 32   3   1   4   2   3   4   0   7  67   0   4   1]
# [  1  11   1   2   0   1   1   0   0  83   0   1   1]
# [  1   1  16   2   0   0   0   0   2  25   1   3   4]
# [  2   2   2  35   1   1   0   0   3  94   1   3   1]
# [  8   5   0   5  10   0   0   0   0  31   1   1   4]
# [  4   2   0   3   0   9   0   0   3  50   0   1   3]
# [  0   0   0   0   0   0   1   0   0   9   0   0   0]
# [  0   0   0   1   1   0   0   0   0 100   0   0   1]
# [  2   2   0   0   1   0   0   0  23  55   1   0   1]
# [  0   3   2   1   0   0   1   0   1 233   0   1   2]
# [  0   2   0   1   0   0   0   0   0  15   7   0   1]
# [  2  11   2   7   1   1   5   0   2 158   1  39   4]
# [  1  16   2   0   0   1   2   1   9 156   1   5  53]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.48225056149269807, 0.3089591567852437, 0.28323511931968515, None)
#**************************************************
#Fold: 1
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [1518 1519 1520 ... 3033 3034 3035]
#training time: 9.26 secs
#predict time: 1.32 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.79      0.34      0.48       122
#   Shopping       0.17      0.10      0.13       106
#  Reference       0.63      0.36      0.46        47
#     Sports       0.57      0.24      0.34       141
#  Computers       0.62      0.16      0.26        62
#       News       0.56      0.17      0.26        59
#      Games       0.01      0.88      0.01         8
#       Home       1.00      0.01      0.02        93
#       Arts       0.57      0.26      0.36       131
#    Society       0.20      0.08      0.11       238
#    Science       0.55      0.30      0.39        20
#   Business       0.68      0.17      0.27       256
#     Health       0.84      0.23      0.36       235
#
#avg / total       0.59      0.19      0.27      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 42   6   0   6   2   0  48   0   4   9   2   1   2]
# [  0  11   0   0   0   0  84   0   2   9   0   0   0]
# [  1   5  17   2   0   0  13   0   0   6   0   1   2]
# [  2   4   2  34   0   0  91   0   0   6   1   1   0]
# [  2   5   0   3  10   1  26   0   3   4   0   5   3]
# [  2   1   0   0   4  10  35   0   1   3   0   2   1]
# [  0   0   0   0   0   0   7   0   0   0   0   1   0]
# [  1   1   1   0   0   0  84   1   1   3   0   0   1]
# [  0   2   0   1   0   0  86   0  34   7   1   0   0]
# [  0   1   1   1   0   1 215   0   0  19   0   0   0]
# [  1   2   0   1   0   1   6   0   0   0   6   3   0]
# [  2  12   2   7   0   5 163   0  10  11   0  43   1]
# [  0  13   4   5   0   0 128   0   5  19   1   6  54]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5892769083876293, 0.18972332015810275, 0.2703066895641696, None)
#**************************************************
#Fold: 2
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [3036 3037 3038 ... 4551 4552 4553]
#training time: 9.29 secs
#predict time: 1.36 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.62      0.21      0.32       122
#   Shopping       0.25      0.13      0.17       102
#  Reference       0.83      0.41      0.55        61
#     Sports       0.42      0.14      0.21       107
#  Computers       0.39      0.11      0.18        61
#       News       0.86      0.18      0.29        68
#      Games       0.00      0.00      0.00         5
#       Home       1.00      0.01      0.02       109
#       Arts       0.57      0.33      0.42       111
#    Society       0.22      0.99      0.35       237
#    Science       0.55      0.29      0.37        21
#   Business       0.73      0.20      0.31       240
#     Health       0.81      0.27      0.41       274
#
#avg / total       0.60      0.33      0.31      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 26   6   0  10   5   1   1   0   4  58   1   7   3]
# [  0  13   0   0   0   0   1   0   2  85   0   0   1]
# [  2   3  25   1   0   0   0   0   2  24   0   3   1]
# [  0   2   0  15   1   0   0   0   1  88   0   0   0]
# [  8   1   0   2   7   0   0   0   2  32   0   3   6]
# [  2   3   0   1   0  12   1   0   1  47   0   0   1]
# [  0   0   0   0   0   0   0   0   0   5   0   0   0]
# [  0   0   0   0   0   0   0   1   1 106   0   0   1]
# [  1   2   0   1   1   0   0   0  37  69   0   0   0]
# [  1   1   0   0   0   0   0   0   1 234   0   0   0]
# [  0   0   0   1   0   0   0   0   1  11   6   0   2]
# [  1   7   0   2   2   1   2   0   7 167   2  47   2]
# [  1  14   5   3   2   0   1   0   6 162   2   4  74]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.6008716645726934, 0.32740447957839264, 0.30838581370624074, None)
#**************************************************
#Fold: 3
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [4554 4555 4556 ... 6069 6070 6071]
#training time: 9.29 secs
#predict time: 1.36 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.77      0.31      0.44       140
#   Shopping       0.25      0.11      0.16       105
#  Reference       0.85      0.39      0.53        44
#     Sports       0.49      0.24      0.32       132
#  Computers       0.47      0.14      0.22        64
#       News       0.70      0.13      0.23        52
#      Games       0.01      0.70      0.01        10
#       Home       1.00      0.01      0.02        89
#       Arts       0.60      0.34      0.43        98
#    Society       0.16      0.07      0.09       256
#    Science       0.55      0.27      0.36        22
#   Business       0.52      0.17      0.26       246
#     Health       0.80      0.25      0.39       260
#
#avg / total       0.55      0.19      0.27      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 43   2   1   9   4   1  58   0   1   9   2   7   3]
# [  2  12   0   1   0   0  78   0   1   6   0   5   0]
# [  0   5  17   1   0   0  13   0   2   1   0   2   3]
# [  1   2   1  32   1   1  79   0   2   9   0   3   1]
# [  4   4   0   3   9   0  27   0   4   2   0   8   3]
# [  0   1   0   2   2   7  29   0   2   5   0   3   1]
# [  1   0   0   0   0   0   7   0   0   0   0   2   0]
# [  0   0   0   0   0   0  83   1   0   4   1   0   0]
# [  0   1   0   2   0   1  55   0  33   6   0   0   0]
# [  1   0   0   2   0   0 235   0   0  17   1   0   0]
# [  0   4   0   2   0   0   8   0   0   1   6   1   0]
# [  2  12   1   7   1   0 155   0   3  16   1  43   5]
# [  2   5   0   4   2   0 137   0   7  29   0   8  66]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5549663575077268, 0.19301712779973648, 0.27084095337028724, None)
#**************************************************
#Fold: 4
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [6072 6073 6074 ... 7587 7588 7589]
#training time: 9.57 secs
#predict time: 1.37 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.63      0.30      0.40       131
#   Shopping       0.19      0.10      0.13        96
#  Reference       0.79      0.45      0.57        60
#     Sports       0.46      0.21      0.29       116
#  Computers       0.41      0.12      0.19        57
#       News       0.64      0.16      0.26        56
#      Games       0.00      0.00      0.00         4
#       Home       0.80      0.04      0.07       103
#       Arts       0.56      0.29      0.38        94
#    Society       0.22      0.98      0.36       238
#    Science       0.53      0.26      0.35        31
#   Business       0.74      0.18      0.29       274
#     Health       0.87      0.26      0.40       258
#
#avg / total       0.59      0.33      0.32      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 39   5   1   7   3   1   1   0   3  68   0   3   0]
# [  2  10   0   0   0   0   1   0   1  81   0   1   0]
# [  1   0  27   1   0   0   0   0   5  23   0   1   2]
# [  2   6   1  24   0   0   0   0   3  78   1   1   0]
# [  9   0   0   5   7   2   0   0   0  26   1   5   2]
# [  2   1   0   3   1   9   0   0   0  37   1   2   0]
# [  0   0   0   0   0   0   0   0   0   4   0   0   0]
# [  0   0   0   1   0   0   0   4   0  96   1   0   1]
# [  0   4   0   1   1   1   0   0  27  60   0   0   0]
# [  0   3   0   0   0   0   0   0   0 233   1   1   0]
# [  2   1   0   0   1   0   0   1   0  16   8   1   1]
# [  3  13   2   7   2   0   5   0   3 185   0  50   4]
# [  2  10   3   3   2   1   4   0   6 154   2   3  68]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5873783448195081, 0.3333333333333333, 0.3179890696020003, None)
#**************************************************
#Fold: 5
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [7590 7591 7592 ... 9105 9106 9107]
#training time: 9.32 secs
#predict time: 1.37 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.63      0.29      0.40       114
#   Shopping       0.13      0.05      0.08       110
#  Reference       0.69      0.38      0.49        53
#     Sports       0.51      0.17      0.25       121
#  Computers       0.47      0.12      0.19        58
#       News       0.73      0.15      0.26        71
#      Games       0.00      0.00      0.00         6
#       Home       0.50      0.01      0.02        97
#       Arts       0.45      0.25      0.32       102
#    Society       0.22      0.96      0.36       251
#    Science       0.58      0.26      0.36        27
#   Business       0.71      0.20      0.31       273
#     Health       0.85      0.26      0.39       235
#
#avg / total       0.54      0.32      0.30      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 33   1   1   5   4   0   3   0   5  55   1   3   3]
# [  1   6   1   0   0   0   1   0   2  96   1   2   0]
# [  2   2  20   2   0   0   0   1   1  20   0   2   3]
# [  4   4   2  20   4   1   0   0   1  81   1   3   0]
# [  4   3   0   4   7   1   1   0   8  27   1   1   1]
# [  3   1   1   1   0  11   0   0   1  49   0   2   2]
# [  0   0   0   0   0   0   0   0   0   6   0   0   0]
# [  0   1   2   0   0   0   0   1   0  93   0   0   0]
# [  0   3   1   1   0   0   0   0  25  72   0   0   0]
# [  1   0   0   1   0   1   2   0   0 242   1   3   0]
# [  1   1   0   0   0   0   0   0   0  16   7   1   1]
# [  2  12   0   3   0   1   5   0   7 188   0  54   1]
# [  1  12   1   2   0   0   4   0   5 145   0   5  60]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5423819465284463, 0.3201581027667984, 0.296800322932342, None)
#**************************************************
#Fold: 6
#Train: [    0     1     2 ... 12141 12142 12143] Validation: [ 9108  9109  9110 ... 10623 10624 10625]
#training time: 9.23 secs
#predict time: 1.38 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.72      0.26      0.38       126
#   Shopping       0.26      0.20      0.23        93
#  Reference       0.75      0.28      0.41        53
#     Sports       0.56      0.23      0.32       131
#  Computers       0.58      0.20      0.30        55
#       News       0.55      0.12      0.19        51
#      Games       0.00      0.00      0.00         5
#       Home       0.00      0.00      0.00        96
#       Arts       0.48      0.29      0.36       104
#    Society       0.23      0.96      0.37       252
#    Science       0.53      0.35      0.42        26
#   Business       0.65      0.16      0.25       287
#     Health       0.76      0.22      0.34       239
#
#avg / total       0.51      0.33      0.30      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 33   6   1   7   3   2   2   0   2  60   2   6   2]
# [  0  19   1   1   0   0   0   0   0  69   1   0   2]
# [  0   3  15   3   0   1   0   0   5  21   0   1   4]
# [  0   6   1  30   0   0   0   0   1  89   1   3   0]
# [  6   3   0   3  11   1   1   0   2  24   2   2   0]
# [  2   1   0   3   1   6   1   0   2  32   0   2   1]
# [  0   0   0   0   0   0   0   0   0   5   0   0   0]
# [  0   0   0   0   0   0   0   0   0  94   0   0   2]
# [  0   4   0   0   1   0   0   0  30  64   0   2   3]
# [  1   3   0   0   1   0   1   0   2 243   1   0   0]
# [  2   1   0   2   0   0   0   0   1   8   9   2   1]
# [  0  15   1   2   0   1   4   0  12 205   0  45   2]
# [  2  11   1   3   2   0   2   0   6 152   1   6  53]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5111892358949021, 0.3254281949934124, 0.3003764701732627, None)
#**************************************************
#Fold: 7
#Train: [    0     1     2 ... 10623 10624 10625] Validation: [10626 10627 10628 ... 12141 12142 12143]
#C:\Users\kennedy\Anaconda3\envs\neuralnet\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)
#training time: 9.34 secs
#predict time: 1.4 secs
#==== CLASSIFICATION REPORT ======
#             precision    recall  f1-score   support
#
# Recreation       0.74      0.33      0.46       118
#   Shopping       0.22      0.11      0.15       110
#  Reference       0.81      0.47      0.59        64
#     Sports       0.51      0.25      0.34       126
#  Computers       0.70      0.12      0.20        60
#       News       0.71      0.15      0.25        67
#      Games       0.17      0.11      0.13         9
#       Home       1.00      0.01      0.02        87
#       Arts       0.54      0.33      0.41       111
#    Society       0.21      0.95      0.35       235
#    Science       0.45      0.20      0.28        25
#   Business       0.62      0.16      0.25       258
#     Health       0.75      0.28      0.41       248
#
#avg / total       0.58      0.33      0.32      1518
#
#**************************************************
#==== CONFUSION MATRIX ======
#[[ 39   0   0  10   0   2   3   0   1  53   0   6   4]
# [  0  12   0   0   1   0   0   0   3  90   0   4   0]
# [  0   1  30   1   0   0   0   0   2  22   1   2   5]
# [  0   4   2  32   0   0   0   0   3  82   0   1   2]
# [  6   3   0   4   7   1   0   0   0  31   0   6   2]
# [  2   2   0   2   0  10   0   0   1  48   1   1   0]
# [  0   0   0   0   0   0   1   0   0   7   0   1   0]
# [  0   1   0   0   0   0   0   1   1  84   0   0   0]
# [  0   2   0   1   0   0   0   0  37  70   0   1   0]
# [  0   3   0   2   0   1   0   0   1 224   1   0   3]
# [  2   2   0   1   0   0   0   0   2  10   5   0   3]
# [  2  11   4   8   1   0   2   0   5 178   3  40   4]
# [  2  14   1   2   1   0   0   0  12 145   0   2  69]]
#**************************************************
#==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======
#(0.5760944496404105, 0.3339920948616601, 0.317613132885128, None)
#**************************************************
#
#CV accuracy: 0.490 +/- 0.062
#**************************************************
#'''