# %%
# First ensemble with NSL-KDD
# Parameters

#----------------------------------------------
# 0 for not using it as base learner
# 1 for using it as base learner

use_model_ada = 1 
use_model_dnn = 1 
use_model_mlp = 1 
use_model_lgbm = 1 
use_model_rf = 1 
use_model_svm = 1
use_model_knn = 1 
#----------------------------------------------
# 0 for training the model
# 1 for using the saved version of the model

load_model_ada = 0 
load_model_dnn = 0 
load_model_mlp = 0 
load_model_lgbm = 0 
load_model_rf = 0 
load_model_svm = 0
load_model_knn = 0 
#----------------------------------------------

# load_model_ada = 1
# load_model_dnn = 1 
# load_model_mlp = 1 
# load_model_lgbm = 1 
# load_model_rf = 1 
# load_model_svm = 1
# load_model_knn = 1 
#----------------------------------------------
feature_selection_bit = 0
# feature_selection_bit = 1




# %%

# Specify the name of the output text file
if feature_selection_bit == 0:
    output_file_name = "ensemble_base_models_all_features_cic.txt"
    with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
    with open(output_file_name, "a") as f: print('---- ensemble_base_models_all_features', file = f)

elif feature_selection_bit == 1:
    output_file_name = "ensemble_base_models_feature_selection_cic.txt"
    with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
    with open(output_file_name, "a") as f: print('----ensemble_base_models_feature_selection--', file = f)


# %%
#!/usr/bin/env python       
# coding: utf-8

# In[1]:
# importing required libraries
import numpy as np
import pandas as pd
import pickle # saving and loading trained model
from os import path


# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense # importing dense layer

from keras.layers import Input
from keras.models import Model
# representation of model layers
#from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import shap

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import time
start_program = time.time()



# %%


def confusion_metrics (name_model,predictions,true_labels):

    name = name_model
    pred_label = predictions
    y_test_01 = true_labels 

    with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print(name, file = f)


    print('---------------------------------------------------------------------------------')
    print('CONFUSION MATRIX')
    print('---------------------------------------------------------------------------------')


    # pred_label = label[ypred]

    confusion_matrix = pd.crosstab(y_test_01, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
    all_unique_values = sorted(set(pred_label) | set(y_test_01))
    z = np.zeros((len(all_unique_values), len(all_unique_values)))
    rows, cols = confusion_matrix.shape
    z[:rows, :cols] = confusion_matrix
    confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
    # confusion_matrix.to_csv('Ensemble_conf_matrix.csv')
    # with open(output_file_name, "a") as f:print(confusion_matrix,file=f)
    print(confusion_matrix)
    with open(output_file_name, "a") as f: print('Confusion Matrix', file = f)

    with open(output_file_name, "a") as f: print(confusion_matrix, file = f)


    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.values.sum() - (FP + FN + TP)
    TP_total = sum(TP)
    TN_total = sum(TN)
    FP_total = sum(FP)
    FN_total = sum(FN)

    TP_total = np.array(TP_total,dtype=np.float64)
    TN_total = np.array(TN_total,dtype=np.float64)
    FP_total = np.array(FP_total,dtype=np.float64)
    FN_total = np.array(FN_total,dtype=np.float64)



    #----------------------------------------------------------------#----------------------------------------------------------------

    print('---------------------------------------------------------------------------------')
    print('METRICS')
    print('---------------------------------------------------------------------------------')


    Acc = accuracy_score(y_test_01, pred_label)
    Precision = precision_score(y_test_01, pred_label, average='macro')
    Recall = recall_score(y_test_01, pred_label, average='macro')
    F1 =  f1_score(y_test_01, pred_label, average='macro')
    BACC = balanced_accuracy_score(y_test_01, pred_label)
    MCC = matthews_corrcoef(y_test_01, pred_label)


    # voting_acc_01 = Acc
    # voting_pre_01 = Precision
    # weighed_avg_rec_01 = Recall
    # weighed_avg_f1_01 = F1
    # weighed_avg_bacc_01 = BACC
    # weighed_avg_mcc_01 = MCC
    # with open(output_file_name, "a") as f:print('Accuracy total: ', Acc,file=f)
    print('Accuracy total: ', Acc)
    print('Precision total: ', Precision )
    print('Recall total: ', Recall )
    print('F1 total: ', F1 )
    print('BACC total: ', BACC)
    print('MCC total: ', MCC)

    with open(output_file_name, "a") as f: print('Accuracy total: ', Acc, file = f)
    with open(output_file_name, "a") as f: print('Precision total: ', Precision, file = f)
    with open(output_file_name, "a") as f: print('Recall total: ', Recall , file = f)
    with open(output_file_name, "a") as f: print('F1 total: ', F1, file = f)
    with open(output_file_name, "a") as f: print('BACC total: ', BACC , file = f)
    with open(output_file_name, "a") as f: print('MCC total: ', MCC, file = f)

    return Acc, Precision, Recall, F1, BACC, MCC


# %%

req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']



# %%

path_str = '/home/oarreche@ads.iu.edu/HITL/cicids/cicids_db/'
fraction = 1
#---------------------------------------------------------------------
#Load Databases from csv file
print('---------------------------------------------------------------------------------')
print('Loading Databases')
print('---------------------------------------------------------------------------------')
print('')


df0 = pd.read_csv (path_str + 'Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)

df1 = pd.read_csv (path_str + 'Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df2 = pd.read_csv (path_str +'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df3 = pd.read_csv (path_str +'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df4 = pd.read_csv (path_str +'Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df5 = pd.read_csv (path_str +'Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df6 = pd.read_csv (path_str +'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df7 = pd.read_csv (path_str +'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


frames = [df0, df1, df2, df3, df4, df5,df6, df7]

df = pd.concat(frames,ignore_index=True)

df = df.sample(frac = 1)
#---------------------------------------------------------------------
# Normalize database

df = pd.concat(frames,ignore_index=True)
df = df.sample(frac = fraction )
y = df.pop(' Label')
df = df.assign(Label = y)

# %%
frac_normal = 0.5

print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')


#filters

filtered_normal = df[df['Label'] == 'BENIGN']

#reduce

reduced_normal = filtered_normal.sample(frac=frac_normal)

#join

df = pd.concat([df[df['Label'] != 'BENIGN'], reduced_normal])

''' ---------------------------------------------------------------'''
df_max_scaled = df.copy()


y = df_max_scaled['Label'].replace({'DoS GoldenEye': 'Dos/Ddos', 'DoS Hulk': 'Dos/Ddos', 'DoS Slowhttptest': 'Dos/Ddos', 'DoS slowloris': 'Dos/Ddos', 'Heartbleed': 'Dos/Ddos', 'DDoS': 'Dos/Ddos','FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force','Web Attack - Brute Force': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack', 'Web Attack - XSS': 'Web Attack'})

df_max_scaled.pop('Label')



# %%
from sklearn.preprocessing import MinMaxScaler
print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df = df_max_scaled.assign( Label = y)
#df
df = df.fillna(0)


y = df.pop('Label')
X = df

# df_max_scaled = df_max_scaled.fillna(0)

# %%
from collections import Counter

counter = Counter(y)
print(counter)
counter_list = list(counter.values())

# %%
# counter['Bot']

# %%
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy={'BENIGN': counter['BENIGN'], 
                                'Dos/Ddos': counter['Dos/Ddos'],
                                'PortScan':counter['PortScan'],
                                'Brute Force':counter['Brute Force'],
                                'Web Attack':counter['Brute Force'],
                                'Bot':counter['Brute Force'],
                                'Infiltration':counter['Brute Force']}, random_state=42)

# smote = SMOTE(sampling_strategy={'BENIGN': 795584, 'Dos/Ddos': 380699,'PortScan':158930,'Brute Force':13835,'Web Attack':13835,'Bot':13835,'Infiltration':13835}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution after oversampling
print("Class distribution after oversampling:", Counter(y_resampled))


# %%
X = X_resampled
y , y_label =pd.factorize(y_resampled)

# y = y_resampled

df = X.assign(Label = y)
# print('train len',counter)

# y = df.pop('Label')

# df = df.assign(Label = y)



# %%
df

# %%
# single_class_train = np.argmax(y_train_multi, axis=1)
# single_class_test = np.argmax(y_test_multi, axis=1)


# df1 = X_train_multi.assign(Label = single_class_train)
# df2 =  X_test_multi.assign(Label = single_class_test)

# frames = [df1,  df2]

# df = pd.concat(frames,ignore_index=True)
# df_fs = df


# %%


# %%
# y = df.pop('Label')
# X = df
# df = X.assign(Label = y)

# %%
# y = df_fs.pop('Label')
# X = df_fs
# df_fs = X.assign(Label = y)


# %% [markdown]
# ## Feature Selection Methods

# %%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_selection import mutual_info_classif

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a decision tree classifier
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Compute information gain using mutual information
# info_gain = mutual_info_classif(X_train, y_train)

# # Display information gain for each feature
# for feature, gain in zip(X_train.columns, info_gain):
#     print(f'Information Gain for {feature}: {gain}')


# %%
if feature_selection_bit == 1 and 0==1:

    from sklearn.feature_selection import mutual_info_classif
    # %matplotlib inline

    # Compute information gain using mutual information
    importances = mutual_info_classif(X, y)

    feat_importances = pd.Series(importances, df.columns[0:len(df.columns)])
    # feat_importances.plot(kind='barh', color = 'teal')
        
    feat_importances_sorted = feat_importances.sort_values( ascending=False)

    # Print or use the sorted DataFrame
    print(feat_importances_sorted)
    # feat_importances_sorted.plot(kind='barh', color = 'teal')
    # feat_importances_sorted
    top_features = feat_importances_sorted.nlargest(10)
    top_feature_names = top_features.index.tolist()

    print("Top 10 feature names:")
    print(top_feature_names)


# %%


# %%

# feat_importances_sorted = feat_importances.sort_values( ascending=False)

# # Print or use the sorted DataFrame
# print(feat_importances_sorted)
# # feat_importances_sorted.plot(kind='barh', color = 'teal')
# # feat_importances_sorted
# top_features = feat_importances_sorted.nlargest(10)
# top_feature_names = top_features.index.tolist()

# print("Top 10 feature names:")
# print(top_feature_names)


# %%
# X

# %%
# from skfeature.function.similarity_based import fisher_score
# import matplotlib.pyplot as plt
# %matplotlib inline 

# ranks = fisher_score.fisher_score(X,y)

# feat_importances = pd.Series(ranks, dataframe.columns[0:len(dataframe.columns)-1])
# feat_importances.plot(kind = 'barh',color = 'teal')
# plt.show()

# %%
# stop

# %%

if feature_selection_bit == 1:
    # USE XAI from last work
    feature_selection = [ 
                    ' Destination Port',
                    ' Init_Win_bytes_backward',
                    ' Packet Length Std',
                    ' Bwd Packet Length Mean', 
                    ' Total Length of Bwd Packets', 
                    ' Packet Length Mean',
                    ' Subflow Bwd Bytes',
                    ' Packet Length Variance', 
                    'Label']


    # Use information gain
    # feature_selection = top_feature_names
    

    df_og = df
    df = df[feature_selection]





# %%

# # y = df.pop('Label')
# # X = df

# y1, y2 = pd.factorize(y)

# y_0 = pd.DataFrame(y1)
# y_1 = pd.DataFrame(y1)
# y_2 = pd.DataFrame(y1)
# y_3 = pd.DataFrame(y1)
# y_4 = pd.DataFrame(y1)


# # y_0 = y_0.replace(0, 0)
# # y_0 = y_0.replace(1, 1)
# y_0 = y_0.replace(2, 1)
# y_0 = y_0.replace(3, 1)
# y_0 = y_0.replace(4, 1)


# y_1 = y_1.replace(1, 999)
# y_1 = y_1.replace(0, 1)
# # y_1 = y_1.replace(1, 0)
# y_1 = y_1.replace(2, 1)
# y_1 = y_1.replace(3, 1)
# y_1 = y_1.replace(4, 1)
# y_1 = y_1.replace(999, 1)


# y_2 = y_2.replace(0, 1)
# y_2 = y_2.replace(1, 1)
# y_2 = y_2.replace(2, 0)
# y_2 = y_2.replace(3, 1)
# y_2 = y_2.replace(4, 1)


# y_3 = y_3.replace(0, 1)
# # y_3 = y_3.replace(1, 1)
# y_3 = y_3.replace(2, 1)
# y_3 = y_3.replace(3, 0)
# y_3 = y_3.replace(4, 1)


# y_4 = y_4.replace(0, 1)
# # y_4 = y_4.replace(1, 1)
# y_4 = y_4.replace(2, 1)
# y_4 = y_4.replace(3, 1)
# y_4 = y_4.replace(4, 0)



# df = df.assign(Label = y)

# %%


#Divide the dataset between level 00 and level 01
import sklearn
from sklearn.model_selection import train_test_split
split = 0.7 

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)

# %%
from collections import Counter

label_counts2 = Counter(y)
print(label_counts2)


# %%
#Base learner Split
# split = 0.7

# X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_00, y_00, train_size=split)

# %%
X_train

# %%
y_train

# %% [markdown]
# ## LEVEL 0 - Weak models - Base Learner

# %%


# %%
with open(output_file_name, "a") as f: print('------------START of WEAK LEARNERS (BASE MODELS) - STACK 00 -----------------', file = f)

#Defining Basemodels


print('---------------------------------------------------------------------------------')
print('Defining RF Model')
print('---------------------------------------------------------------------------------')
#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)
#------------------------------------------------------------------------------


print('---------------------------------------------------------------------------------')
print('Defining ADA Model')
print('---------------------------------------------------------------------------------')
#ADA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
import time
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)


print('---------------------------------------------------------------------------------')
print('Defining LGBM Model')
print('---------------------------------------------------------------------------------')
#LGBM
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()



#KNN
print('---------------------------------------------------------------------------------')
print('Defining KNN Model')
print('---------------------------------------------------------------------------------')
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors = 5)


#SVM
print('---------------------------------------------------------------------------------')
print('Defining SVM Model')
print('---------------------------------------------------------------------------------')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier

# Instantiate the SGDClassifier with additional hyperparameters
clf = SGDClassifier(
    loss='hinge',           # hinge loss for linear SVM
    penalty='l2',           # L2 regularization to prevent overfitting
    alpha=1e-4,             # Learning rate (small value for fine-grained updates)
    max_iter=1000,          # Number of passes over the training data
    random_state=42,        # Seed for reproducible results
    learning_rate='optimal' # Automatically adjusts the learning rate based on the training data
)


#MLP
print('---------------------------------------------------------------------------------')
print('Defining MLP Model')
print('---------------------------------------------------------------------------------')


from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import time

# create MLPClassifier instance
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)


#DNN
print('---------------------------------------------------------------------------------')
print('Defining DNN Model')
print('---------------------------------------------------------------------------------')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# #Model Parameters
# dropout_rate = 0.01
# nodes = 70
# out_layer = 5
# optimizer='adam'
# loss='sparse_categorical_crossentropy'
# epochs=1
# batch_size=2*256

#Model Parameters
dropout_rate = 0.2
nodes = 3
out_layer = 7
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=100
batch_size=128


num_columns = X_train.shape[1]

dnn = tf.keras.Sequential()

# Input layer
dnn.add(tf.keras.Input(shape=(num_columns,)))

# Dense layers with dropout
dnn.add(tf.keras.layers.Dense(nodes))
dnn.add(tf.keras.layers.Dropout(dropout_rate))

dnn.add(tf.keras.layers.Dense(nodes))
dnn.add(tf.keras.layers.Dropout(dropout_rate))

dnn.add(tf.keras.layers.Dense(nodes))
dnn.add(tf.keras.layers.Dropout(dropout_rate))

dnn.add(tf.keras.layers.Dense(nodes))
dnn.add(tf.keras.layers.Dropout(dropout_rate))

dnn.add(tf.keras.layers.Dense(nodes))
dnn.add(tf.keras.layers.Dropout(dropout_rate))

# Output layer
dnn.add(tf.keras.layers.Dense(out_layer, activation='softmax'))

dnn.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

dnn.summary()



# dnn = Sequential()
# dnn.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Input layer
# dnn.add(Dense(64, activation='relu'))  # Hidden layer
# dnn.add(Dense(5))  # Output layer

# dnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # summary of model layers
# dnn.summary()

# %%


# %%
# #SVM
# # Wrap SGDClassifier with MultiOutputClassifier
# multi_target_clf = MultiOutputClassifier(clf)

# # Fit the model on the training data
# multi_target_clf.fit(X_train, y_train)

# Make predictions on the test data
# y_pred = clf.predict(X_test)



# %%
#Training Basemodels
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
n_splits = 5  # You can adjust the number of folds as needed



print('---------------------------------------------------------------------------------')
print('Training Model')
with open(output_file_name, "a") as f: print('Training weak models - level 0', file = f)

print('---------------------------------------------------------------------------------')

if use_model_ada == 1 and load_model_ada == 0:

    print('---------------------------------------------------------------------------------')
    print('Training ADA')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training ADA', file = f)
    print('---------------------------------------------------------------------------------')
    #ADA


    start = time.time()
    ada = abc.fit(X_train, y_train)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(ada, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    ada_tr_time_taken= time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)

    # Assuming 'model' is your trained model
    # joblib.dump(ada, 'ada_base_model.joblib')


if use_model_rf == 1 and load_model_rf == 0:

    print('---------------------------------------------------------------------------------')
    print('Training RF')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
    with open(output_file_name, "a") as f: print('Training RF', file = f)
    print('---------------------------------------------------------------------------------')
    #RF
    start = time.time()
    model_rf = rf.fit(X_train,y_train)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(model_rf, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    rf_tr_time_taken = time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(model_rf, 'rf_base_model.joblib')

if use_model_svm == 1 and load_model_svm == 0:

    print('---------------------------------------------------------------------------------')
    print('Training SVM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training SVM', file = f)
    print('---------------------------------------------------------------------------------')
    #SVM

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    # clf.score(X_train, y_train)
    svm_tr_time_taken= time_taken = end - start

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(clf, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(clf, 'svm_base_model.joblib')


if use_model_knn == 1 and load_model_knn == 0:

    #KNN
    print('---------------------------------------------------------------------------------')
    print('Training KNN')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training KNN', file = f)
    print('---------------------------------------------------------------------------------')
    start = time.time()
    knn_clf.fit(X_train,y_train)
    end = time.time()


    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(knn_clf, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    knn_tr_time_taken = time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(knn_clf, 'knn_base_model.joblib')


if use_model_lgbm == 1 and load_model_lgbm == 0:


    print('---------------------------------------------------------------------------------')
    print('Training LGBM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training LGBM', file = f)
    print('---------------------------------------------------------------------------------')
    start = time.time()
    lgbm.fit(X_train, y_train)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(lgbm, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    lgbm_tr_time_taken = time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(lgbm, 'lgbm_base_model.joblib')

if use_model_mlp == 1 and load_model_mlp == 0:


    print('---------------------------------------------------------------------------------')
    print('Training MLP')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training MLP', file = f)
    print('---------------------------------------------------------------------------------')

    start = time.time()
    MLP = mlp.fit(X_train, y_train)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(MLP, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    mlp_tr_time_taken= time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(MLP, 'mlp_base_model.joblib')


if use_model_dnn == 1 and load_model_dnn == 0:
    from keras.callbacks import EarlyStopping

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    print('---------------------------------------------------------------------------------')
    print('Training DNN')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training DNN', file = f)
    print('---------------------------------------------------------------------------------')
    # Convert Y_test back to its original format
    # y_test = np.argmax(Y_test, axis=1)

    # Start the timer
    start = time.time()
    # dnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    dnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2, callbacks=[early_stopping])

    # End the timer
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(dnn, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    dnn_tr_time_taken= time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # dnn.save("DNN_base_model.h5")

    # Calculate the time taken and print it out
    # print(f'Time taken for training: {time_taken} seconds')


with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)



# %%
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# # Define your Keras model as a function
# def create_model(optimizer='adam', hidden_layer_size=16):
#     # model = Sequential()
#     # model.add(Dense(hidden_layer_size, input_dim=input_size, activation='relu'))
#     # model.add(Dense(1, activation='sigmoid'))
#     # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        
#     dnn = tf.keras.Sequential()

#     # Input layer
#     dnn.add(tf.keras.Input(shape=(num_columns,)))

#     # Dense layers with dropout
#     dnn.add(tf.keras.layers.Dense(nodes))
#     dnn.add(tf.keras.layers.Dropout(dropout_rate))

#     dnn.add(tf.keras.layers.Dense(nodes))
#     dnn.add(tf.keras.layers.Dropout(dropout_rate))

#     dnn.add(tf.keras.layers.Dense(nodes))
#     dnn.add(tf.keras.layers.Dropout(dropout_rate))

#     dnn.add(tf.keras.layers.Dense(nodes))
#     dnn.add(tf.keras.layers.Dropout(dropout_rate))

#     dnn.add(tf.keras.layers.Dense(nodes))
#     dnn.add(tf.keras.layers.Dropout(dropout_rate))

#     # Output layer
#     dnn.add(tf.keras.layers.Dense(out_layer))



#     dnn.compile(optimizer=optimizer, loss=loss)

#     dnn.summary()
#     return dnn

# # Create a KerasClassifier
# dnn = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'optimizer': ['adam', 'sgd'],
#     'hidden_layer_size': [8, 16, 32]
# }

# # Create the StratifiedKFold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Create GridSearchCV
# grid = GridSearchCV(estimator=dnn, param_grid=param_grid, cv=cv, scoring='accuracy')
# grid_result = grid.fit(X_train, y_train)

# # Print the best parameters and best accuracy
# print("Best Parameters: ", grid_result.best_params_)
# print("Best Accuracy: ", grid_result.best_score_)



# %%
# stratified_kfold

# %%
# Loading Models
from tensorflow.keras.models import load_model

if load_model_ada == 1:
    ada = joblib.load('ada_base_model.joblib')

if load_model_svm == 1:
    clf =  joblib.load('svm_base_model.joblib')

if load_model_dnn == 1:
    dnn = load_model("DNN_base_model.h5")

if load_model_knn == 1:
    knn_clf = joblib.load('knn_base_model.joblib')

if load_model_mlp == 1:
    MLP = joblib.load('mlp_base_model.joblib')

if load_model_rf == 1:
    rf = joblib.load('rf_base_model.joblib')

if load_model_lgbm == 1:
    lgbm = joblib.load('lgbm_base_model.joblib')







# %%
# Make predictions on the test data
# preds_svm = clf.predict(X_test)



# y_scores = y_pred
# y_true = y_test



# %% [markdown]
# ### Base leaners predictions

# %%
from sklearn.calibration import CalibratedClassifierCV
with open(output_file_name, "a") as f: print('Generating Predictions', file = f)

if use_model_rf == 1:

    print('---------------------------------------------------------------------------------')
    print('Prediction RF')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction RF', file = f)
    print('---------------------------------------------------------------------------------')
    #RF
    start = time.time()
    preds_rf = rf.predict(X_test)
    preds_rf_prob = rf.predict_proba(X_test)
    end = time.time()
    rf_pr_time_taken=  time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

if use_model_svm == 1:

    print('Prediction SVM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction SVM', file = f)
    print('---------------------------------------------------------------------------------')
    #SVM
    start = time.time()
    preds_svm = clf.predict(X_test)
    # preds_svm_prob = clf.predict_proba(X_test)

    #Since SVM does not deal with prob by nature we use a meta learner
    # https://stackoverflow.com/questions/55250963/how-to-get-probabilities-for-sgdclassifier-linearsvm

    model = CalibratedClassifierCV(clf)

    model.fit(X, y)
    preds_svm_prob = model.predict_proba(X)

    end = time.time()
    svm_pr_time_taken = time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

if use_model_lgbm == 1:

    print('Prediction LGBM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction LGBM', file = f)
    print('---------------------------------------------------------------------------------')
    #LGBM
    start = time.time()
    preds_lgbm = lgbm.predict(X_test)
    preds_lgbm_prob = lgbm.predict_proba(X_test)

    end = time.time()
    lgbm_pr_time_taken=time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

if use_model_dnn == 1:

    print('Prediction DNN')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction DNN', file = f)
    print('---------------------------------------------------------------------------------')
    #DNN
    start = time.time()
    pred_dnn = dnn.predict(X_test)
    preds_dnn_prob = pred_dnn
    preds_dnn = np.argmax(pred_dnn,axis = 1)
    end = time.time()
    dnn_pr_time_taken=time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

if use_model_ada == 1:

    print('Prediction ADA')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction ADA', file = f)
    print('---------------------------------------------------------------------------------')
    #ADA
    start = time.time()
    preds_ada = ada.predict(X_test)
    preds_ada_prob = ada.predict_proba(X_test)

    end = time.time()
    ada_pr_time_taken=time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')
    print('Prediction MLP')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction MLP', file = f)
    print('---------------------------------------------------------------------------------')

if use_model_mlp == 1:

    #MLP
    start = time.time()
    y_pred = MLP.predict_proba(X_test)
    preds_mlp_prob = y_pred
    preds_mlp = np.argmax(y_pred,axis = 1)
    end = time.time()
    mlp_pr_time_taken=time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')
    print('Prediction KNN')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction KNN', file = f)
    print('---------------------------------------------------------------------------------')

if use_model_knn == 1:

    #KNN
    start = time.time()
    preds_knn =knn_clf.predict(X_test)
    preds_knn_prob =knn_clf.predict_proba(X_test)

    preds_knn
    end = time.time()
    knn_pr_time_taken=time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)


# %%
# from sklearn.calibration import CalibratedClassifierCV
# model = CalibratedClassifierCV(clf)

# model.fit(X, y)
# preds_svm_prob = model.predict_proba(X)

# print(preds_ada_prob)
# print(preds_knn_prob)
# print(preds_dnn_prob)
# print(preds_mlp_prob)
# print(preds_rf_prob)
# print(preds_svm_prob)


# %%
print(preds_svm_prob)
preds_3 = np.argmax(preds_svm_prob,axis = 1)
print(preds_3)

print(preds_svm)
# print(y_train)

# %% [markdown]
# ### METRICS - Base Learners

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score



# >>> 
# >>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
# 0.99...
# >>> roc_auc_score(y, clf.decision_function(X))

# %% [markdown]
# #### RF

# %%
# y_test
# pred_label

# %%
#RF
if use_model_rf == 1:
    # start = time.time()
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('RF base model', file = f)

    pred_label = preds_rf
    name = 'rf'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    globals()[f"{name}_time_00"] = rf_pr_time_taken + rf_tr_time_taken


# %%
#DNN
if use_model_dnn == 1:
    start = time.time()
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('DNN base model', file = f)


    pred_label = preds_dnn
    name = 'dnn'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = dnn_pr_time_taken + dnn_tr_time_taken

# %%
#ADA
if use_model_ada == 1:
    start = time.time()
    
    pred_label = preds_ada
    name = 'ada'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = ada_pr_time_taken + ada_tr_time_taken

# %%
#SVM
if use_model_svm == 1:
    start = time.time()

    pred_label = preds_svm
    name = 'svm'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = svm_pr_time_taken + svm_tr_time_taken

# %%
#KNN
if use_model_knn == 1:
    start = time.time()
    pred_label = preds_knn
    name = 'knn'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = knn_pr_time_taken + knn_tr_time_taken

# %%
#MLP
if use_model_mlp == 1:
    start = time.time()
    pred_label = preds_mlp
    name = 'mlp'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = mlp_pr_time_taken + mlp_tr_time_taken

# %%
#lgbm
start_lgbm = time.time()
if use_model_lgbm == 1:

    pred_label = preds_lgbm
    name = 'lgbm'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start_lgbm
    globals()[f"{name}_time_00"] = lgbm_pr_time_taken + lgbm_tr_time_taken

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
start = time.time()

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)
# Make predictions on the test data
preds_dt = dt_classifier.predict(X_test)
# Evaluate the accuracy of the model
preds_dt_prob = dt_classifier.predict_proba(X_test)


pred_label = preds_dt
name = 'dt'
metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    

globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ### CATBOOST

# %%
import catboost
start = time.time()
cat_00 = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')

# Fit the model
cat_00.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=10)

# Make predictions on the test set
preds_cat = cat_00.predict(X_test)
preds_cat_prob = cat_00.predict_proba(X_test)
preds_cat = np.squeeze(preds_cat)


if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Catboost base model', file = f)


    

    pred_label = preds_cat
    
    

    # pred_label = y_pred

    name = 'cat'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = time_taken


# %%
import xgboost as xgb
start = time.time()
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# Assuming you have your features and labels as X and y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': 7,  # specify the number of classes
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'  # metric for multi-class classification
}

# Train the XGBoost model
num_round = 100
xgb_00 = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
preds_xgb = xgb_00.predict(dtest)
# preds_xgb_prob = xgb_00.predict_proba(dtest)


# Get class probabilities
# Assuming binary classification, get the probability for the positive class (class 1)
preds_xgb_margin = xgb_00.predict(dtest, output_margin=True)
preds_xgb_prob = 1 / (1 + np.exp(-preds_xgb_margin))

# Print or use positive_class_probabilities as needed
# print(positive_class_probabilities)


# Convert predicted probabilities to class labels (if necessary)
# y_pred_labels = [round(value) for value in y_pred]

# Evaluate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


# %%

if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('xgboost base model', file = f)




    pred_label = preds_xgb
    # pred_label = label[ypred]
    name = 'xgb'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ### LR

# %%
from sklearn.linear_model import LogisticRegression

#Logistic Regression
print('---------------------------------------------------------------------------------')
print('Defining Logistic Regression Model')
print('---------------------------------------------------------------------------------')
logreg_00 = LogisticRegression()

if 1 == 1 and 0 == 0:
    print('---------------------------------------------------------------------------------')
    print('Training LR ')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training LR', file = f)
    print('---------------------------------------------------------------------------------')
    start_lr = start = time.time()
    logreg_00.fit(X_train,y_train)
    end = time.time()


    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(knn_clf, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(logreg_01, 'logreg_01.joblib')


# if 1 == 1:
    # logreg_01 = joblib.load('logreg_01.joblib')

if 1 == 1:

    #lR
    start = time.time()
    preds_lr = preds_logreg =logreg_00.predict(X_test)
    preds_lr_prob = logreg_00.predict_proba(X_test)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

#LR
if 1 == 1:
    pred_label = preds_logreg
    name = 'lr'
    metrics = confusion_metrics(name, pred_label, y_test)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    globals()[f"{name}_acc_00"] = Acc
    globals()[f"{name}_pre_00"] = Precision
    globals()[f"{name}_rec_00"] = Recall
    globals()[f"{name}_f1_00"] = F1
    globals()[f"{name}_bacc_00"] = BACC
    globals()[f"{name}_mcc_00"] = MCC
    
    end = time.time()
    time_taken = end - start_lr
    globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ### Bagging DT  
# 

# %%
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
start = time.time()
# # Define the base classifier (Decision Tree in this case)
base_classifier = DecisionTreeClassifier(random_state=42)

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)


pred_label = y_pred
name = 'bag_dt'
metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    

globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ## bagging SVM

# %%
## bagging  with SVM
from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

start = time.time()

from sklearn.linear_model import SGDClassifier

# Instantiate the SGDClassifier with additional hyperparameters
svm_01 = SGDClassifier(
    loss='hinge',           # hinge loss for linear SVM
    penalty='l2',           # L2 regularization to prevent overfitting
    alpha=1e-4,             # Learning rate (small value for fine-grained updates)
    max_iter=1000,          # Number of passes over the training data
    random_state=42,        # Seed for reproducible results
    learning_rate='optimal' # Automatically adjusts the learning rate based on the training data
)

# # Define the base classifier (Decision Tree in this case)
base_classifier = svm_01

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)


with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_svm'
pred_label = y_pred
metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC

end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ## Bagging MLP

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import time
start = time.time()
# create MLPClassifier instance
mlp_00 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)

base_classifier = mlp_00

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_00, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_mlp'
pred_label = y_pred
metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken


# %% [markdown]
# ### bagging KNN

# %%
from sklearn.neighbors import KNeighborsClassifier
knn_00=KNeighborsClassifier(n_neighbors = 5)
start = time.time()
base_classifier = knn_00

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_00, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_knn'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# 
# ### bag LogRegression
# 

# %%
from sklearn.linear_model import LogisticRegression
start = time.time()
#Logistic Regression
print('---------------------------------------------------------------------------------')
print('Defining baggin Logistic Regression Model')
print('---------------------------------------------------------------------------------')
logreg_00 = LogisticRegression()


base_classifier = logreg_00

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_00, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_lr'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken

# %% [markdown]
# ### Bagging RF

# %%
start = time.time()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)

base_classifier = rf

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_rf'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken



# %%


# %% [markdown]
# ### Bagging ADA

# %%
start = time.time()

from sklearn.ensemble import AdaBoostClassifier
import time
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

base_classifier = ada

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_ada'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken


# %%


# %% [markdown]
# ### Bagging LGBM

# %%
start = time.time()

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()


base_classifier = lgbm

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_lgbm'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken



# %%


# %% [markdown]
# ### Bagging Catboost 

# %%
import catboost
start = time.time()

bag_cat = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')

base_classifier = bag_cat

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_cat'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken



# %%


# %% [markdown]
# ### Bagging Combined

# %%
### Bagging with many models
##### do bootstrapping 
##### 1. Multiple subsets are created from the original dataset, selecting observations with replacement.

start = time.time()

num_bootstraps = 10  # Adjust the number of bootstraps as needed

original_data_df = X_train.assign(label = y_train)
boot_df = []
for i in range(0,num_bootstraps): 
    boot_df.append(original_data_df.sample(frac = 1, replace=True).reset_index(drop=True))

# boot_df[5]

#### 2.A base model (weak model) is created on each of these subsets.
bag_comb_pred = []

# SVM
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(
    loss='hinge',           # hinge loss for linear SVM
    penalty='l2',           # L2 regularization to prevent overfitting
    alpha=1e-4,             # Learning rate (small value for fine-grained updates)
    max_iter=1000,          # Number of passes over the training data
    random_state=42,        # Seed for reproducible results
    learning_rate='optimal' # Automatically adjusts the learning rate based on the training data
)
y_train_boot = boot_df[0].pop('label')
X_train_boot = boot_df[0]
clf.fit(X_train_boot, y_train_boot)
preds_svm_00 = clf.predict(X_test)
bag_comb_pred.append(preds_svm_00)




#ADA
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
ada = abc.fit(X_train, y_train)
y_train_boot = boot_df[1].pop('label')
X_train_boot = boot_df[1]
preds_ada_00 = ada.predict(X_test)
bag_comb_pred.append(preds_ada_00)

#Catboost
import catboost
cat_00 = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')
y_train_boot = boot_df[2].pop('label')
X_train_boot = boot_df[2]
cat_00.fit(X_train_boot, y_train_boot, eval_set=(X_test, y_test), verbose=10)
preds_cat = cat_00.predict(X_test)
preds_cat = np.squeeze(preds_cat)
pred_label = preds_cat
bag_comb_pred.append(preds_cat)

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)
y_train_boot = boot_df[3].pop('label')
X_train_boot = boot_df[3]
if 1 == 1 and 0 == 0:
    MLP = mlp.fit(X_train_boot, y_train_boot)
    y_pred = MLP.predict_proba(X_test)
    preds_mlp_00 = np.argmax(y_pred,axis = 1)

bag_comb_pred.append(preds_mlp_00)

#LGBM
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
y_train_boot = boot_df[4].pop('label')
X_train_boot = boot_df[4]

if 1 == 1 and 0 == 0:
    lgbm.fit(X_train_boot, y_train_boot)
    preds_lgbm_00 = lgbm.predict(X_test)
    bag_comb_pred.append(preds_lgbm_00)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf_00=KNeighborsClassifier(n_neighbors = 5)
y_train_boot = boot_df[5].pop('label')
X_train_boot = boot_df[5]

if 1 == 1 and 0 == 0:
    knn_clf_00.fit(X_train_boot,y_train_boot)
if use_model_knn == 1:
    preds_knn =knn_clf_00.predict(X_test)
    bag_comb_pred.append(preds_knn)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)
y_train_boot = boot_df[6].pop('label')
X_train_boot = boot_df[6]

if True == True:
    model_rf_00 = rf.fit(X_train_boot,y_train_boot)
    preds_rf_00 = model_rf_00.predict(X_test)
    bag_comb_pred.append(preds_rf_00)
#DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#Model Parameters
y_train_boot = boot_df[7].pop('label')
X_train_boot = boot_df[7]


dropout_rate = 0.2
nodes = 3
out_layer = 7
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=100
batch_size=128
num_columns = X_train_boot.shape[1]
dnn_00 = tf.keras.Sequential()
# Input layer
dnn_00.add(tf.keras.Input(shape=(num_columns,)))
# Dense layers with dropout
dnn_00.add(tf.keras.layers.Dense(nodes))
dnn_00.add(tf.keras.layers.Dropout(dropout_rate))
dnn_00.add(tf.keras.layers.Dense(nodes))
dnn_00.add(tf.keras.layers.Dropout(dropout_rate))
dnn_00.add(tf.keras.layers.Dense(nodes))
dnn_00.add(tf.keras.layers.Dropout(dropout_rate))
dnn_00.add(tf.keras.layers.Dense(nodes))
dnn_00.add(tf.keras.layers.Dropout(dropout_rate))
dnn_00.add(tf.keras.layers.Dense(nodes))
dnn_00.add(tf.keras.layers.Dropout(dropout_rate))
# Output layer
# dnn_00.add(tf.keras.layers.Dense(out_layer))
dnn_00.add(tf.keras.layers.Dense(out_layer, activation='softmax'))
dnn_00.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
from keras.callbacks import EarlyStopping
# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
dnn_00.fit(X_train_boot, y_train_boot, epochs=epochs, batch_size=batch_size,validation_split=0.2, callbacks=[early_stopping])
pred_dnn = dnn_00.predict(X_test)
preds_dnn_00 = np.argmax(pred_dnn,axis = 1)
bag_comb_pred.append(preds_dnn_00)
#LogReg
from sklearn.linear_model import LogisticRegression
logreg_00 = LogisticRegression()
y_train_boot = boot_df[8].pop('label')
X_train_boot = boot_df[8]

logreg_00.fit(X_train_boot,y_train_boot)
preds_logreg =logreg_00.predict(X_test)
bag_comb_pred.append(preds_logreg)
import xgboost as xgb
y_train_boot = boot_df[9].pop('label')
X_train_boot = boot_df[9]

# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_boot, label=y_train_boot)
dtest = xgb.DMatrix(X_test, label=y_test)
# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': 7,  # specify the number of classes
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'  # metric for multi-class classification
}
# Train the XGBoost model
num_round = 100
xgb_00 = xgb.train(params, dtrain, num_round)
preds_xgb_00 = xgb_00.predict(dtest)
bag_comb_pred.append(preds_xgb_00)
### 3. The models run in parallel and are independent of each other.
bag_vot_df = pd.DataFrame()
for i in range(0,len(bag_comb_pred)):
    bag_vot_df[f'model_{i}'] =  bag_comb_pred[i]
print(bag_vot_df)
# Voting start
from scipy.stats import mode
# bag_comb_pred_df = pd.DataFrame(bag_comb_pred)
# Extract predictions columns

# predictions = df[['dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm','cat','xgb']]
    # selected_columns = df.loc[:, ~df.columns.isin(['rf'])]
predictions = bag_vot_df 

# predictions = bag_comb_pred_df.loc[:, ~df.columns.isin(['label'])] #df[column_features]

# Use the mode function along axis 1 to get the most common prediction for each row
ensemble_predictions, _ = mode(predictions.values, axis=1)

# Add the ensemble predictions to the DataFrame
bag_vot_df['ensemble'] = ensemble_predictions.astype(int)

# Display the DataFrame with ensemble predictions
print(bag_vot_df)

pred_label = bag_vot_df ['ensemble'].values
bag_vot_df.pop('ensemble')


name='bag_comb'
metrics = confusion_metrics(name, pred_label, y_test)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_00"] = Acc
globals()[f"{name}_pre_00"] = Precision
globals()[f"{name}_rec_00"] = Recall
globals()[f"{name}_f1_00"] = F1
globals()[f"{name}_bacc_00"] = BACC
globals()[f"{name}_mcc_00"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_00"] = time_taken



# %%


# %%


# %% [markdown]
# ## Creating new dataset for level 01

# %%
print(len(preds_dnn_prob), len(y_test))

# %%
print(y_test)

# %%
y_test = pd.DataFrame(y_test)
df_from_series = y_test
y_test_reset_index = df_from_series.reset_index()
# y_test2 = y_test.reset_index(inplace=True)
print(y_test_reset_index)
y_test_reset_index.pop('index')

# %%
y_test_reset_index.values[0][0]

# %%


# %%
preds_dnn_2 = []
preds_svm_2 = []
preds_rf_2 = []
preds_mlp_2 = []
preds_ada_2 = []
preds_knn_2 = []
preds_lgbm_2 = []
preds_cat_2 = []
preds_xgb_2 = []

preds_lr_2 = []
preds_dt_2 = []

for i in range(0,len(preds_dnn_prob)):  
    # print(i)
    # print(preds_dnn_prob[i][y_test_reset_index.values[i][0]])
    preds_dnn_2.append(preds_dnn_prob[i][y_test_reset_index.values[i][0]])
    preds_svm_2.append(preds_svm_prob[i][y_test_reset_index.values[i][0]])
    preds_rf_2.append(preds_rf_prob[i][y_test_reset_index.values[i][0]])
    preds_mlp_2.append(preds_mlp_prob[i][y_test_reset_index.values[i][0]])
    preds_ada_2.append(preds_ada_prob[i][y_test_reset_index.values[i][0]])
    preds_knn_2.append(preds_knn_prob[i][y_test_reset_index.values[i][0]])
    preds_lgbm_2.append(preds_lgbm_prob[i][y_test_reset_index.values[i][0]])
    preds_cat_2.append(preds_cat_prob[i][y_test_reset_index.values[i][0]])
    preds_xgb_2.append(preds_xgb_prob[i][y_test_reset_index.values[i][0]])
    preds_lr_2.append(preds_lr_prob[i][y_test_reset_index.values[i][0]])
    preds_dt_2.append(preds_dt_prob[i][y_test_reset_index.values[i][0]])

    

# %%


# %%
with open(output_file_name, "a") as f: print('------------------------------------------------------------------', file = f)
with open(output_file_name, "a") as f: print('------------------------------------------------------------------', file = f)
with open(output_file_name, "a") as f: print('------------------------------------------------------------------', file = f)

with open(output_file_name, "a") as f: print('------------START of STRONGER LEARNER - STACK 01 -----------------', file = f)


# Stack the vectors horizontally to create a matrix
column_features = ['dnn','rf','lgbm','ada','knn','mlp','svm','cat','xgb','lr','dt','label']
training_matrix2 = np.column_stack((
                          preds_dnn_2,
                          preds_rf_2,
                          preds_lgbm_2,
                          preds_ada_2,
                          preds_knn_2, 
                          preds_mlp_2,
                          preds_svm_2,
                          preds_cat_2,
                          preds_xgb_2,
                          preds_lr_2,
                          preds_dt_2,
                          y_test
                          ))

training_matrix = np.column_stack((
                          preds_dnn,
                          preds_rf,
                          preds_lgbm,
                          preds_ada,
                          preds_knn, 
                          preds_mlp,
                          preds_svm,
                          preds_cat,
                          preds_xgb,
                          preds_lr,
                          preds_dt,
                        #   preds
                          y_test
                          ))
# Print the resulting matrix
print(training_matrix)
print(training_matrix2)

# %%
df_level_00_0 = pd.DataFrame(training_matrix, columns=column_features)
df_level_00_1 = pd.DataFrame(training_matrix2, columns=column_features)

# %%

# Assuming df is your DataFrame
if feature_selection_bit == 1:

    df_level_00_1.to_csv('base_models_prob_feature_selection.csv', index=False)
    df_level_00_0.to_csv('base_models_class_feature_selection.csv', index=False)
    
if feature_selection_bit == 0:

    df_level_00_1.to_csv('base_models_prob_all_features.csv', index=False)
    df_level_00_0.to_csv('base_models_class_all_features.csv', index=False)

# %%
df_level_00_1


# %%
df_level_00_0

# %%
# y_01 = df_level_01.pop('label')
# X_01 = df_level_01
# df_level_01 = df_level_01.assign(label = y_01)

# %%
# X_01

# %%
# y_01

# %%
# df_level_01

# %%
# split = 0.7

# X_train_01,X_test_01, y_train_01, y_test_01 = sklearn.model_selection.train_test_split(X_01, y_01, train_size=split)

# %%


# %%
# from tabulate import tabulate

# # Assuming data is a 110x4 list, where each row is a sublist
# # data =  [["Row {} Col {}".format(i + 1, j + 1) for j in range(4)] for i in range(110)]
# data = [["" for _ in range(3)] for _ in range(12)]

# # Manually insert data at specific row and column
# # data[0][0] = "ADA"
# # data[1][0] = "DNN"
# # data[2][0] = "SVM"
# # data[3][0] = "ADA"
# # data[4][0] = "DNN"
# # data[2][0] = "SVM"


# names_models = ['ADA',
#                 'SVM',
#                 'DNN',
#                 'MLP',
#                 'KNN',
#                 'CAT',
#                 'XGB',
#                 'LGBM',
#                 'RF',
#                 'LR',
#                 'VOTING'
#                 ]
# level_00_f1 = [ada_f1_00,
#                 svm_f1_00,
#                 dnn_f1_00,
#                 mlp_f1_00,
#                 knn_f1_00,
#                 cat_f1_00,
#                 xgb_f1_00,
#                 lgbm_f1_00,
#                 rf_f1_00,
#                 lr_f1_00,
#                 voting_f1_00]  

                 

# for i in range(0,len(names_models)):
#     data[i][0] =  names_models[i]
#     data[i][1] = level_00_f1[i]
#     data[i][2] = level_01_f1[i]


 
# # data[0][1] = ada_acc_00
# # data

# # Define column headers
# headers = ["F1", "Level 00", "Level 01"]

# # Print the table
# table = tabulate(data, headers=headers, tablefmt="grid")
# print(table)
# with open(output_file_name, "a") as f: print(table, file = f)


# %%
# lr_acc_00 = 0 
# voting_acc_00 = 0

# lr_pre_00 = 0 
# voting_pre_00 = 0

# lr_rec_00 = 0 
# voting_rec_00 = 0

# lr_f1_00 = 0 
# voting_f1_00 = 0

# %%
from tabulate import tabulate

# Assuming data is a 110x4 list, where each row is a sublist
# data =  [["Row {} Col {}".format(i + 1, j + 1) for j in range(4)] for i in range(110)]
data = [["" for _ in range(5)] for _ in range(24)]

# Manually insert data at specific row and column
# data[0][0] = "ADA"
# data[1][0] = "DNN"
# data[2][0] = "SVM"
# data[3][0] = "ADA"
# data[4][0] = "DNN"
# data[2][0] = "SVM"


# names_models = ['ADA',
#                 'SVM',
#                 'DNN',
#                 'MLP',
#                 'KNN',
#                 'CAT',
#                 'XGB',
#                 'LGBM',
#                 'RF',
#                 'LR',
#                 'VOTING',
#                 '   '
#                 ]

# names_models = ['ADA',
#                 'SVM',
#                 'DNN',
#                 'MLP',
#                 'KNN',
#                 'CAT',
#                 'XGB',
#                 'LGBM',
#                 'RF',
#                 'LR',
#                 'DT',
#                 # 'VOTING',
#                 'Bag_svm',
#                 'Bag_knn',
#                 'Bag_DT',
#                 'Bag_LR',
#                 'Bag_mlp',
#                 # 'avg',
#                 # 'weighed_avg'
#                 ]

names_models = ['ADA',
                'SVM',
                'DNN',
                'MLP',
                'KNN',
                'CAT',
                'XGB',
                'LGBM',
                'RF',
                'LR',
                'DT',
                # 'VOTING',
                'Bag_svm',
                'Bag_knn',
                'Bag_DT',
                'Bag_LR',
                'Bag_mlp',

                'Bag_rf',
                'Bag_ada',
                'Bag_lgbm',
                # 'Bag_xgb',
                'Bag_cat',
                'Bag_comb',

                # 'avg',
                # 'weighed_avg'
                ]


level_00_acc = [ada_acc_00,
                svm_acc_00,
                dnn_acc_00,
                mlp_acc_00,
                knn_acc_00,
                cat_acc_00,
                xgb_acc_00,
                lgbm_acc_00,
                rf_acc_00,
                lr_acc_00,
                dt_acc_00,
                # voting_acc_00,
                bag_svm_acc_00,
                bag_knn_acc_00,
                bag_dt_acc_00,
                bag_lr_acc_00,
                bag_mlp_acc_00,
               
                bag_rf_acc_00,
                bag_ada_acc_00,
                bag_lgbm_acc_00,

                bag_cat_acc_00,
                bag_comb_acc_00,
               
               
                
                # avg_acc_00,
                # weighed_avg_acc_00
                ]  

                # ]  

level_00_pre = [ada_pre_00,
                svm_pre_00,
                dnn_pre_00,
                mlp_pre_00,
                knn_pre_00,
                cat_pre_00,
                xgb_pre_00,
                lgbm_pre_00,
                rf_pre_00,
                lr_pre_00,
                dt_pre_00,
                # voting_pre_00,
                bag_svm_pre_00,
                bag_knn_pre_00,
                bag_dt_pre_00,
                bag_lr_pre_00,
                bag_mlp_pre_00,

                bag_rf_pre_00,
                bag_ada_pre_00,
                bag_lgbm_pre_00,

                bag_cat_pre_00,
                bag_comb_pre_00,
               
                # avg_pre_00,
                # weighed_avg_pre_00
                ]  

level_00_rec = [ada_rec_00,
                svm_rec_00,
                dnn_rec_00,
                mlp_rec_00,
                knn_rec_00,
                cat_rec_00,
                xgb_rec_00,
                lgbm_rec_00,
                rf_rec_00,
                lr_rec_00,
                dt_rec_00,
                # voting_rec_00,
                bag_svm_rec_00,
                bag_knn_rec_00,
                bag_dt_rec_00,
                bag_lr_rec_00,
                bag_mlp_rec_00,

                bag_rf_rec_00,
                bag_ada_rec_00,
                bag_lgbm_rec_00,

                bag_cat_rec_00,
                bag_comb_rec_00,
               
                # avg_rec_00,
                # weighed_avg_rec_00
                ]  

level_00_f1 = [ada_f1_00,
                svm_f1_00,
                dnn_f1_00,
                mlp_f1_00,
                knn_f1_00,
                cat_f1_00,
                xgb_f1_00,
                lgbm_f1_00,
                rf_f1_00,
                lr_f1_00,
                dt_rec_00,
                # voting_f1_00,
                bag_svm_f1_00,
                bag_knn_f1_00,
                bag_dt_f1_00,
                bag_lr_f1_00,
                bag_mlp_f1_00,

                bag_rf_f1_00,
                bag_ada_f1_00,
                bag_lgbm_f1_00,

                bag_cat_f1_00,
                bag_comb_f1_00,
               
                # avg_f1_00,
                # weighed_avg_f1_00
                ]                   

for i in range(0,len(names_models)):
    data[i][0] =  names_models[i]

    data[i][1] = level_00_acc[i]
    # data[i][2] = level_01_acc[i]

    data[i][2] = level_00_pre[i] 
    # data[i][4] = level_01_pre[i]

    data[i][3] = level_00_rec[i] 
    # data[i][6] = level_01_rec[i]

    data[i][4] = level_00_f1[i]
    # data[i][8] = level_01_f1[i]




 
# data[0][1] = ada_acc_00
# data

# Define column headers
# headers = ["Models", "ACC-00", " ACC-01","PRE-00", " PRE-01","REC-00", " REC-01","F1-00", " F1-01",]
headers = ["Models", "ACC-00","PRE-00","REC-00","F1-00"]


# Print the table
table = tabulate(data, headers=headers, tablefmt="grid")
print(table)
# with open(output_file_name, "a") as f: print(table, file = f)


# %%
# Combine data into a list of tuples for sorting
model_data = list(zip(names_models, level_00_acc, level_00_pre, level_00_rec, level_00_f1))

# Sort by F1-00 score in descending order
model_data_sorted = sorted(model_data, key=lambda x: x[4], reverse=True)

# Separate the sorted data back into individual lists
sorted_names_models, sorted_level_00_acc, sorted_level_00_pre, sorted_level_00_rec, sorted_level_00_f1 = zip(*model_data_sorted)

# Assign the sorted data to the table
for i in range(len(sorted_names_models)):
    data[i][0] = sorted_names_models[i]
    data[i][1] = sorted_level_00_acc[i]
    data[i][2] = sorted_level_00_pre[i] 
    data[i][3] = sorted_level_00_rec[i] 
    data[i][4] = sorted_level_00_f1[i]

# Define column headers
headers = ["Models", "ACC-00", "PRE-00", "REC-00", "F1-00"]

# Print the table
table = tabulate(data, headers=headers, tablefmt="grid")
with open(output_file_name, "a") as f: print('Summary table - LEVEL 00', file = f)

if feature_selection_bit == 1: 
    with open(output_file_name, "a") as f: print('Feature Selection was applied', file = f)
else:
    with open(output_file_name, "a") as f: print('All features were used', file = f)


    
print(table)
with open(output_file_name, "a") as f: print(table, file = f)

# %%
# implement time table
from tabulate import tabulate

names_models = ['ADA',
                'SVM',
                'DNN',
                'MLP',
                'KNN',
                'CAT',
                'XGB',
                'LGBM',
                'RF',
                'LR',
                'DT',
                # 'VOTING',
                'Bag_svm',
                'Bag_knn',
                'Bag_DT',
                'Bag_LR',
                'Bag_mlp',

                'Bag_rf',
                'Bag_ada',
                'Bag_lgbm',
                # 'Bag_xgb',
                'Bag_cat',
                'Bag_comb',
                # 'avg',
                # 'weighed_avg'
                ]

data = [["" for _ in range(2)] for _ in range(len(names_models))]

level_00_time = [
                ada_time_00,
                svm_time_00,
                dnn_time_00,
                mlp_time_00,
                knn_time_00,
                cat_time_00,
                xgb_time_00,
                lgbm_time_00,
                rf_time_00,
                lr_time_00,
                dt_time_00,
                # voting_time_00,
                bag_svm_time_00,
                bag_knn_time_00,
                bag_dt_time_00,
                bag_lr_time_00,
                bag_mlp_time_00,

                bag_rf_time_00,
                bag_ada_time_00,
                bag_lgbm_time_00,
                # bag_xgb_time_00,
                bag_cat_time_00,
                bag_comb_time_00,

                # avg_time_00,
                # weighed_avg_time_00
                ]  


# Combine data into a list of tuples for sorting
model_data = list(zip(names_models, level_00_time))

# Sort by F1-00 score in descending order
model_data_sorted = sorted(model_data, key=lambda x: x[1], reverse=False)

# Separate the sorted data back into individual lists
sorted_names_models, sorted_level_00_time = zip(*model_data_sorted)

# Assign the sorted data to the table
for i in range(len(sorted_names_models)):
    data[i][0] = sorted_names_models[i]
    data[i][1] = sorted_level_00_time[i]

# Define column headers
headers = ["Models", "time-00(sec)"]


# Print the table
table = tabulate(data, headers=headers, tablefmt="grid")
with open(output_file_name, "a") as f: print('Time is counted is seconds', file = f)
print(table)
with open(output_file_name, "a") as f: print(table, file = f)
end_program = time.time()
time_program = end_program - start_program
with open(output_file_name, "a") as f: print('Running time of entire program is:', time_program ,' seconds',file = f)


