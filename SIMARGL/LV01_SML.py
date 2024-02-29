# %%
# First ensemble with NSL-KDD
# Parameters
# Few parameters are not fully implemented yet

#----------------------------------------------
# 0 for not using it as base learner
# 1 for using it as base learner
# not implemented but in the code in someparts
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

# load_model_ada = 0 
# load_model_dnn = 0 
# load_model_mlp = 0 
# load_model_lgbm = 0 
# load_model_rf = 0 
# load_model_svm = 0
# load_model_knn = 0 
#----------------------------------------------
# not implemented but in the code in someparts
load_model_ada = 1
load_model_dnn = 1 
load_model_mlp = 1 
load_model_lgbm = 1 
load_model_rf = 1                               
load_model_svm = 1
load_model_knn = 1 
#----------------------------------------------

# Implemented
#----------------------------------------------
feature_selection_bit = 0 # OFF
# feature_selection_bit = 1 # On
# pick_prob = 1 # set equal one to choose the dataset with probabilities, set to 0 to choose one with the classes.
pick_prob = 1
generate_feature_importance = 0 # Generate Shap graphs


column_features = [
                    'dnn',
                   'rf',
                   'lgbm',
                   'ada',
                #    'knn',
                #    'mlp',
                   'svm',
                   'cat',
                   'xgb',
                   'lr',
                   'dt',
                   'label']


# %%
# Specify the name of the output text file
if feature_selection_bit == 0:

    if pick_prob == 0:
        output_file_name = "ensemble_level_01_all_features_classes.txt"
        with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
        with open(output_file_name, "a") as f: print('----ensemble_level_01_all_features_classes--', file = f)

    elif pick_prob == 1:
        output_file_name = "ensemble_level_01_all_features_probabilites.txt"
        with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
        with open(output_file_name, "a") as f: print('----ensemble_level_01_all_features_probabilites--', file = f)

elif feature_selection_bit == 1:
    if pick_prob == 0:
        output_file_name = "ensemble_level_01_feature_selection_classes.txt"
        with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
        with open(output_file_name, "a") as f: print('----ensemble_level_01_feature_selection_classes--', file = f)
    elif pick_prob == 1:
        output_file_name = "ensemble_level_01_feature_selection_probabilites.txt"
        with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
        with open(output_file_name, "a") as f: print('----ensemble_level_01_feature_selection_probabilites--', file = f)

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
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

import joblib
from sklearn.model_selection import train_test_split
import sklearn
from tabulate import tabulate



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

import time
start_program = time.time()

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

import joblib
from sklearn.model_selection import train_test_split
import sklearn
from tabulate import tabulate



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

df_level_00_1=pd.read_csv('base_models_prob_feature_selection.csv')
df_level_00_0=pd.read_csv('base_models_class_feature_selection.csv')


df_level_00_1
y1 = df_level_00_1.pop('label')
X1 = df_level_00_1
df_level_00_1 = X1.assign(label = y1)
y0 = df_level_00_0.pop('label')
X0 = df_level_00_0
df_level_00_0 = X0.assign(label = y0)

if feature_selection_bit == 1:

    from sklearn.feature_selection import mutual_info_classif
    # %matplotlib inline

    # Compute information gain using mutual information
    importances0 = mutual_info_classif(X0, y0)
    importances1 = mutual_info_classif(X1, y1)


    feat_importances0 = pd.Series(importances0, df_level_00_0.columns[0:len(df_level_00_0.columns)-1])
    feat_importances1= pd.Series(importances1, df_level_00_1.columns[0:len(df_level_00_1.columns)-1])

    # feat_importances.plot(kind='barh', color = 'teal')
        
    feat_importances_sorted0 = feat_importances0.sort_values( ascending=False)
    feat_importances_sorted1 = feat_importances1.sort_values( ascending=False)


    # Print or use the sorted DataFrame
    print(feat_importances_sorted0)
    print(feat_importances_sorted1)

    # feat_importances_sorted.plot(kind='barh', color = 'teal')
    # feat_importances_sorted
    top_features0 = feat_importances_sorted0.nlargest(5)
    top_features1 = feat_importances_sorted1.nlargest(5)

    top_feature_names0 = top_features0.index.tolist()
    top_feature_names1 = top_features1.index.tolist()


    print("Top 5 feature names:")
    print(top_feature_names0)
    print(top_feature_names1)

    column_features0 = top_feature_names0
    column_features1 = top_feature_names1

    # df_level_00_0 = df_level_00_0[column_features0]
    # df_level_00_1 = df_level_00_1[column_features1]



# %%


# %%

# Assuming df is your DataFrame
# if feature_selection_bit == 1:
#     df_level_00_1=pd.read_csv('base_models_prob_feature_selection.csv',names=column_features)
#     df_level_00_0=pd.read_csv('base_models_class_feature_selection.csv',names=column_features)

# if feature_selection_bit == 0:

#     df_level_00_1=pd.read_csv('base_models_prob_feature_selection.csv')
#     df_level_00_0=pd.read_csv('base_models_class_feature_selection.csv')

df_level_00_1=pd.read_csv('base_models_prob_feature_selection.csv')
df_level_00_0=pd.read_csv('base_models_class_feature_selection.csv')

if feature_selection_bit == 1:
    df_level_00_0 = df_level_00_0[column_features0]
    df_level_00_1 = df_level_00_1[column_features1]


# %%
df_level_00_1


# %%
df_level_00_0

# %%
y1

# %%
if pick_prob == 1:
    df_level_01 = df_level_00_1
else: 
    df_level_01 = df_level_00_0

df_level_01 = df_level_01.assign(label = y1)

y_01 = df_level_01.pop('label')
X_01 = df_level_01
df_level_01 = df_level_01.assign(label = y_01)


split = 0.7
X_train_01,X_test_01, y_train_01, y_test_01 = sklearn.model_selection.train_test_split(X_01, y_01, train_size=split)

# %%


# df_level_02 = pd.read_csv('base_models_class_feature_selection.csv')

# df_level_02

# y_02 = df_level_02.pop('label')
# X_02 = df_level_02
# df_level_02 = df_level_02.assign(label = y_01)


# split = 0.7
# X_train_02,X_test_02, y_train_02, y_test_02 = sklearn.model_selection.train_test_split(X_02, y_02, train_size=split)

# %% [markdown]
# ## Training the stronger model - STACK level 01

# %%
#----------------------------------------------------------------
with open(output_file_name, "a") as f: print('Stack model - Strong learner - level 01', file = f)
with open(output_file_name, "a") as f: print('-------------------------------------------------------', file = f)

# %%
X_test_01

# %%


# %%


# %% [markdown]
# ### Voting

# %%
start = time.time()
    
if pick_prob == 0:
    # Voting start

    import pandas as pd
    from scipy.stats import mode

    # Assuming 'df' is your original DataFrame with columns 'dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm', 'label'
    df = X_test_01
    # Extract predictions columns
    
    # predictions = df[['dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm','cat','xgb']]
        # selected_columns = df.loc[:, ~df.columns.isin(['rf'])]
    predictions = df.loc[:, ~df.columns.isin(['label'])] #df[column_features]

    # Use the mode function along axis 1 to get the most common prediction for each row
    ensemble_predictions, _ = mode(predictions.values, axis=1)

    # Add the ensemble predictions to the DataFrame
    df['ensemble'] = ensemble_predictions.astype(int)

    # Display the DataFrame with ensemble predictions
    print(df)

    pred_label = df ['ensemble'].values
    df.pop('ensemble')

    #testing metrics def
    name = 'voting'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    end = time.time()
    time_taken = end - start

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    globals()[f"{name}_time_01"] = time_taken
   
else:
    name = 'voting'
    globals()[f"{name}_acc_01"] = 0
    globals()[f"{name}_pre_01"] = 0
    globals()[f"{name}_rec_01"] = 0
    globals()[f"{name}_f1_01"] = 0
    globals()[f"{name}_bacc_01"] = 0
    globals()[f"{name}_mcc_01"] = 0
    globals()[f"{name}_time_01"] = 9999
   

# %%
voting_acc_01


# %% [markdown]
# ### Average

# %%
start = time.time()

# if pick_prob == 0:
if 0 == 0:
    # Average start

    import pandas as pd
    from scipy.stats import mode

    # Assuming 'df' is your original DataFrame with columns 'dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm', 'label'
    df = X_test_01
    predictions = df.loc[:, ~df.columns.isin(['label'])] #df[column_features]

   

    column_sums = df.sum(axis=1)
    row_average = df.mean(axis=1)

    # Approximate the result to the closest integer
    rounded_average = row_average.round().astype(int)

    # print(rounded_average)

    df['results'] = rounded_average
    print(df)
 
    pred_label = df ['results'].values

    # pred_label = df ['ensemble'].values
    # df.pop('ensemble')
    df.pop('results')

    # df.pop('column_sums')

    with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

    name = 'avg'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    end = time.time()
    time_taken = end - start
    
    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    globals()[f"{name}_time_01"] = time_taken

    

# %% [markdown]
# ## Weighed Average

# %%
# column_features
# move this up with column_features

#important update this as you need to select the important features,
#  the left is the least important while the right is the most important
# needs automation
if pick_prob == 1 and feature_selection_bit == 1:
    column_features = column_features1

elif pick_prob == 0 and feature_selection_bit == 1: 
    column_features = column_features0

else: None

feature_selection_columns_in_order_of_importance = column_features[:-1]



# %%
start = time.time()

# if pick_prob == 0:
if 0 == 0:
    # Average start

    import pandas as pd
    from scipy.stats import mode

    # Assuming 'df' is your original DataFrame with columns 'dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm', 'label'
    # df = X_test_01
    df = X_test_01[feature_selection_columns_in_order_of_importance]
    # Extract predictions columns
    
    # predictions = df[['dnn', 'rf', 'lgbm', 'ada', 'knn', 'mlp', 'svm','cat','xgb']]
        # selected_columns = df.loc[:, ~df.columns.isin(['rf'])]
    predictions = df.loc[:, ~df.columns.isin(['label'])] #df[column_features]

    # weight
    weights_values = []

    # linear weight distribution
    for i in range(0,len(~df.columns.isin(['label']))):
        weights_values.append(i/(len(~df.columns.isin(['label']))-1))
    print(weights_values)
    # weights_values = [10,3,2,2.3]
    print(weights_values)
    print(df)
    weighted_average = df.multiply(weights_values).sum(axis=1) / sum(weights_values)
    print(weighted_average)
    # Approximate the result to the closest integer
    rounded_weighted_average = weighted_average.round().astype(int)

    print(rounded_weighted_average)

    # print(rounded_average)

    df['results'] = rounded_weighted_average
    print(df)
 
    pred_label = df ['results'].values

    # pred_label = df ['ensemble'].values
    # df.pop('ensemble')
    df.pop('results')

    # df.pop('column_sums')

    #testing metrics def
    name = 'weighed_avg'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    

    end = time.time()
    time_taken = end - start
    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    globals()[f"{name}_time_01"] = time_taken
    
    


# %% [markdown]
# ## bagging  with DT

# %%
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

start = time.time()
base_classifier = DecisionTreeClassifier(random_state=42)

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_01, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_dt'
pred_label = y_pred
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    

end = time.time()
time_taken = end - start
globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ## bagging  with SVM
# 

# %%


# %%
## bagging  with SVM
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier

start = time.time()

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
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)


with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_svm'
pred_label = y_pred
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC

end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ## bagging with DNN

# %%
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# #Model Parameters
# dropout_rate = 0.2
# nodes = 3
# out_layer = 5
# optimizer='adam'
# loss='sparse_categorical_crossentropy'
# epochs=100
# batch_size=128


# num_columns = X_train_01.shape[1]

# dnn_01 = tf.keras.Sequential()

# # Input layer
# dnn_01.add(tf.keras.Input(shape=(num_columns,)))

# # Dense layers with dropout
# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# # Output layer
# # dnn_01.add(tf.keras.layers.Dense(out_layer))

# dnn_01.add(tf.keras.layers.Dense(out_layer, activation='softmax'))


# dnn_01.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

# base_classifier = dnn_01

# # Define the BaggingClassifier
# bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# # Train the BaggingClassifier
# bagging_classifier.fit(X_train_01, y_train_01)

# # Make predictions on the test set
# y_pred = bagging_classifier.predict(X_test_01)

# # Evaluate accuracy
# # accuracy = accuracy_score(y_test_01, y_pred)
# # print(f'Accuracy: {accuracy}')

# with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

# with open(output_file_name, "a") as f: print('Bagging with DNN', file = f)


# print('---------------------------------------------------------------------------------')
# print('CONFUSION MATRIX')
# print('---------------------------------------------------------------------------------')


# pred_label = y_pred

# confusion_matrix = pd.crosstab(y_test_01, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
# all_unique_values = sorted(set(pred_label) | set(y_test_01))
# z = np.zeros((len(all_unique_values), len(all_unique_values)))
# rows, cols = confusion_matrix.shape
# z[:rows, :cols] = confusion_matrix
# confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
# # confusion_matrix.to_csv('Ensemble_conf_matrix.csv')
# # with open(output_file_name, "a") as f:print(confusion_matrix,file=f)
# print(confusion_matrix)
# with open(output_file_name, "a") as f: print('Confusion Matrix', file = f)

# with open(output_file_name, "a") as f: print(confusion_matrix, file = f)


# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
# TN = confusion_matrix.values.sum() - (FP + FN + TP)
# TP_total = sum(TP)
# TN_total = sum(TN)
# FP_total = sum(FP)
# FN_total = sum(FN)

# TP_total = np.array(TP_total,dtype=np.float64)
# TN_total = np.array(TN_total,dtype=np.float64)
# FP_total = np.array(FP_total,dtype=np.float64)
# FN_total = np.array(FN_total,dtype=np.float64)



# #----------------------------------------------------------------#----------------------------------------------------------------

# print('---------------------------------------------------------------------------------')
# print('METRICS')
# print('---------------------------------------------------------------------------------')


# Acc = accuracy_score(y_test_01, pred_label)
# Precision = precision_score(y_test_01, pred_label, average='macro')
# Recall = recall_score(y_test_01, pred_label, average='macro')
# F1 =  f1_score(y_test_01, pred_label, average='macro')
# BACC = balanced_accuracy_score(y_test_01, pred_label)
# MCC = matthews_corrcoef(y_test_01, pred_label)


# bag_dnn_acc_01 = Acc
# bag_dnn_pre_01 = Precision
# bag_dnn_rec_01 = Recall
# bag_dnn_f1_01 = F1
# bag_dnn_bacc_01 = BACC
# bag_dnn_mcc_01 = MCC
# # with open(output_file_name, "a") as f:print('Accuracy total: ', Acc,file=f)
# print('Accuracy total: ', Acc)
# print('Precision total: ', Precision )
# print('Recall total: ', Recall )
# print('F1 total: ', F1 )
# print('BACC total: ', BACC)
# print('MCC total: ', MCC)

# with open(output_file_name, "a") as f: print('Accuracy total: ', Acc, file = f)
# with open(output_file_name, "a") as f: print('Precision total: ', Precision, file = f)
# with open(output_file_name, "a") as f: print('Recall total: ', Recall , file = f)
# with open(output_file_name, "a") as f: print('F1 total: ', F1, file = f)
# with open(output_file_name, "a") as f: print('BACC total: ', BACC , file = f)
# with open(output_file_name, "a") as f: print('MCC total: ', MCC, file = f)


# %% [markdown]
# ## bagging with MLP

# %%


# %%
from sklearn.neural_network import MLPClassifier
start = time.time()

# create MLPClassifier instance
mlp_01 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)

base_classifier = mlp_01

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_01, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_mlp'
pred_label = y_pred
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken



# %%


# %% [markdown]
# ## bagging knn

# %%


# %%
from sklearn.neighbors import KNeighborsClassifier
knn_01=KNeighborsClassifier(n_neighbors = 5)
start = time.time()

base_classifier = knn_01

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_01, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_knn'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ## bagging LogRegression

# %%
from sklearn.linear_model import LogisticRegression
start = time.time()

#Logistic Regression
print('---------------------------------------------------------------------------------')
print('Defining baggin Logistic Regression Model')
print('---------------------------------------------------------------------------------')
logreg_01 = LogisticRegression()


base_classifier = logreg_01

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

# Evaluate accuracy
# accuracy = accuracy_score(y_test_01, y_pred)
# print(f'Accuracy: {accuracy}')

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_lr'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


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
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_ada'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### Bagging CAT

# %%
import catboost
start = time.time()

bag_cat = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')

base_classifier = bag_cat

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_cat'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken



# %% [markdown]
# ### Baggin LGBM
# 

# %%
start = time.time()

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()


base_classifier = lgbm

# Define the BaggingClassifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the BaggingClassifier
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_lgbm'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken



# %% [markdown]
# ### Bagging XGB

# %%

# import xgboost as xgb

# # Create a DMatrix for XGBoost
# dtrain = xgb.DMatrix(X_train_01, label=y_train_01)
# dtest = xgb.DMatrix(X_test_01, label=y_test_01)

# # Set XGBoost parameters
# params = {
#     'objective': 'multi:softmax',  # for multi-class classification
#     'num_class': 5,  # specify the number of classes
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'eval_metric': 'mlogloss'  # metric for multi-class classification
# }

# # Train the XGBoost model
# num_round = 100
# xgb_01 = xgb.train(params, dtrain, num_round)

# base_classifier = xgb

# # Define the BaggingClassifier
# bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# # Train the BaggingClassifier
# bagging_classifier.fit(X_train_01, y_train_01)

# # Make predictions on the test set
# y_pred = bagging_classifier.predict(X_test_01)

# with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

# name = 'bag_xgb'

# pred_label = y_pred


# metrics = confusion_metrics(name, pred_label, y_test_01)

# Acc = metrics[0]
# Precision = metrics[1]
# Recall = metrics[2]
# F1 = metrics[3]
# BACC = metrics[4]
# MCC = metrics[5]    


# globals()[f"{name}_acc_01"] = Acc
# globals()[f"{name}_pre_01"] = Precision
# globals()[f"{name}_rec_01"] = Recall
# globals()[f"{name}_f1_01"] = F1
# globals()[f"{name}_bacc_01"] = BACC
# globals()[f"{name}_mcc_01"] = MCC



# %%


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
bagging_classifier.fit(X_train_01, y_train_01)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test_01)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

name = 'bag_rf'

pred_label = y_pred


metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken



# %%


# %%


# %% [markdown]
# ### Bagging with many models

# %% [markdown]
# ##### do bootstrapping 

# %% [markdown]
# ##### 1. Multiple subsets are created from the original dataset, selecting observations with replacement.
# 

# %%
start = time.time()

num_bootstraps = 10  # Adjust the number of bootstraps as needed

original_data_df = X_train_01.assign(label = y_train_01)

# %%
boot_df = []
for i in range(0,num_bootstraps): 
    boot_df.append(original_data_df.sample(frac = 1, replace=True).reset_index(drop=True))


# %%
boot_df[5]

# %% [markdown]
# #### 2.A base model (weak model) is created on each of these subsets.

# %%
bag_comb_pred = []


# %%
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
preds_svm_01 = clf.predict(X_test_01)
bag_comb_pred.append(preds_svm_01)





# %%
#ADA
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
ada = abc.fit(X_train_01, y_train_01)
y_train_boot = boot_df[1].pop('label')
X_train_boot = boot_df[1]
preds_ada_01 = ada.predict(X_test_01)
bag_comb_pred.append(preds_ada_01)


# %%
#Catboost
import catboost
cat_01 = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')
y_train_boot = boot_df[2].pop('label')
X_train_boot = boot_df[2]
cat_01.fit(X_train_boot, y_train_boot, eval_set=(X_test_01, y_test_01), verbose=10)
preds_cat = cat_01.predict(X_test_01)
preds_cat = np.squeeze(preds_cat)
pred_label = preds_cat
bag_comb_pred.append(preds_cat)


# %%
#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)
y_train_boot = boot_df[3].pop('label')
X_train_boot = boot_df[3]
if 1 == 1 and 0 == 0:
    MLP = mlp.fit(X_train_boot, y_train_boot)
    y_pred = MLP.predict_proba(X_test_01)
    preds_mlp_01 = np.argmax(y_pred,axis = 1)

bag_comb_pred.append(preds_mlp_01)


# %%
#LGBM
print('---------------------------------------------------------------------------------')
print('Defining LGBM Model')
print('---------------------------------------------------------------------------------')
#LGBM
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
y_train_boot = boot_df[4].pop('label')
X_train_boot = boot_df[4]

if 1 == 1 and 0 == 0:
    lgbm.fit(X_train_boot, y_train_boot)
    preds_lgbm_01 = lgbm.predict(X_test_01)
    bag_comb_pred.append(preds_lgbm_01)

# %%
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf_01=KNeighborsClassifier(n_neighbors = 5)
y_train_boot = boot_df[5].pop('label')
X_train_boot = boot_df[5]

if 1 == 1 and 0 == 0:
    knn_clf_01.fit(X_train_boot,y_train_boot)
if use_model_knn == 1:
    preds_knn =knn_clf_01.predict(X_test_01)
    bag_comb_pred.append(preds_knn)

# %%
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)
y_train_boot = boot_df[6].pop('label')
X_train_boot = boot_df[6]

if True == True:
    model_rf_01 = rf.fit(X_train_boot,y_train_boot)
    preds_rf_01 = model_rf_01.predict(X_test_01)
    bag_comb_pred.append(preds_rf_01)

# %%
#DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#Model Parameters
y_train_boot = boot_df[7].pop('label')
X_train_boot = boot_df[7]


dropout_rate = 0.02
nodes = 3
out_layer = 5
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=100
batch_size=128
num_columns = X_train_boot.shape[1]
dnn_01 = tf.keras.Sequential()
# Input layer
dnn_01.add(tf.keras.Input(shape=(num_columns,)))
# Dense layers with dropout
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))
# Output layer
# dnn_01.add(tf.keras.layers.Dense(out_layer))
dnn_01.add(tf.keras.layers.Dense(out_layer, activation='softmax'))
# dnn.add(tf.keras.layers.Dense(out_layer, activation='softmax'))
dnn_01.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
from keras.callbacks import EarlyStopping
# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
dnn_01.fit(X_train_boot, y_train_boot, epochs=epochs, batch_size=batch_size,validation_split=0.2, callbacks=[early_stopping])
pred_dnn = dnn_01.predict(X_test_01)
preds_dnn_01 = np.argmax(pred_dnn,axis = 1)
bag_comb_pred.append(preds_dnn_01)

# %%
#LogReg
from sklearn.linear_model import LogisticRegression
logreg_01 = LogisticRegression()
y_train_boot = boot_df[8].pop('label')
X_train_boot = boot_df[8]

logreg_01.fit(X_train_boot,y_train_boot)
preds_logreg =logreg_01.predict(X_test_01)
bag_comb_pred.append(preds_logreg)

# %%
import xgboost as xgb
y_train_boot = boot_df[9].pop('label')
X_train_boot = boot_df[9]

# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_boot, label=y_train_boot)
dtest = xgb.DMatrix(X_test_01, label=y_test_01)
# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': 5,  # specify the number of classes
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'  # metric for multi-class classification
}
# Train the XGBoost model
num_round = 100
xgb_01 = xgb.train(params, dtrain, num_round)
preds_xgb_01 = xgb_01.predict(dtest)
bag_comb_pred.append(preds_xgb_01)

# %% [markdown]
# ### 3. The models run in parallel and are independent of each other.

# %%
bag_vot_df = pd.DataFrame()
for i in range(0,len(bag_comb_pred)):
    bag_vot_df[f'model_{i}'] =  bag_comb_pred[i]
print(bag_vot_df)

# %%
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



# %%
name='bag_comb'
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### DNN

# %%
print('---------------------------------------------------------------------------------')
print('Defining DNN Model')
print('---------------------------------------------------------------------------------')
start_dnn = time.time()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Model Parameters
dropout_rate = 0.2
nodes = 3
out_layer = 5
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=100
batch_size=128


num_columns = X_train_01.shape[1]

dnn_01 = tf.keras.Sequential()

# Input layer
dnn_01.add(tf.keras.Input(shape=(num_columns,)))

# # Dense layers with dropout
# dnn_01.add(tf.keras.layers.Dense(nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(2*nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(3*nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn_01.add(tf.keras.layers.Dense(2*nodes))
# dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))



# Dense layers with dropout
dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

dnn_01.add(tf.keras.layers.Dense(nodes))
dnn_01.add(tf.keras.layers.Dropout(dropout_rate))

# Output layer
# dnn_01.add(tf.keras.layers.Dense(out_layer))

dnn_01.add(tf.keras.layers.Dense(out_layer, activation='softmax'))
# dnn.add(tf.keras.layers.Dense(out_layer, activation='softmax'))


dnn_01.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

dnn_01.summary()



# %%
#DNN
try:
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
    # dnn_01.fit(X_train_01, y_train_01, epochs=epochs, batch_size=batch_size)
    dnn_01.fit(X_train_01, y_train_01, epochs=epochs, batch_size=batch_size,validation_split=0.2, callbacks=[early_stopping])

    # model.fit(x_train, Y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

    # End the timer
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    # joblib.dump(dnn_01, 'dnn_level_01.joblib')
    # dnn_01.save("dnn_level_01.h5")

    # Calculate the time taken and print it out
    # print(f'Time taken for training: {time_taken} seconds')
except: 
    None

# %%
# dnn_01 = load_model("dnn_level_01.h5")


# %%
#DNN
try:
    start = time.time()
    pred_dnn = dnn_01.predict(X_test_01)
    preds_dnn_01 = np.argmax(pred_dnn,axis = 1)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
except:
        with open(output_file_name, "a") as f: print('error', file = f)
        preds_dnn_01 = 0


# %%
try:
    name = 'dnn'
    pred_label = preds_dnn_01
        
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start_dnn
    globals()[f"{name}_time_01"] = time_taken

except: None    

# %% [markdown]
# ### SVM

# %%
#SVM
print('---------------------------------------------------------------------------------')
print('Defining SVM Model')
print('---------------------------------------------------------------------------------')
start_svm = time.time()

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

#SVM
start = time.time()
clf.fit(X_train_01, y_train_01)
end = time.time()
clf.score(X_train_01, y_train_01)
time_taken = end - start
with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
joblib.dump(clf, 'svm_level_01.joblib')


clf = loaded_model = joblib.load('svm_level_01.joblib')


#SVM
start = time.time()
preds_svm_01 = clf.predict(X_test_01)
end = time.time()
time_taken = end - start
with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
print('---------------------------------------------------------------------------------')



# %%

pred_label = preds_svm_01
name = 'svm'
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start_svm
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### Random Forest

# %%

print('---------------------------------------------------------------------------------')
print('Defining RF Model')
print('---------------------------------------------------------------------------------')
start_rf = time.time()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)
#------------------------------------------------------------------------------

if True == True:

    print('---------------------------------------------------------------------------------')
    print('Training RF')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
    with open(output_file_name, "a") as f: print('Training RF', file = f)
    print('---------------------------------------------------------------------------------')
    #RF
    start = time.time()
    model_rf_01 = rf.fit(X_train_01,y_train_01)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(model_rf_01, X_train_01, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)


    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    joblib.dump(model_rf_01, 'rf_base_model_01.joblib')

if 1 == 1:
    model_rf_01  = joblib.load('rf_base_model_01.joblib')

if 1 == 1:

    print('---------------------------------------------------------------------------------')
    print('Prediction RF')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction RF', file = f)
    print('---------------------------------------------------------------------------------')
    #RF
    start = time.time()
    preds_rf_01 = model_rf_01.predict(X_test_01)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

    with open(output_file_name, "a") as f: print('-------------------------------------------------------', file = f)
pred_label = preds_rf_01
name='rf'
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start_rf
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### LGBM

# %%
print('---------------------------------------------------------------------------------')
print('Defining LGBM Model')
print('---------------------------------------------------------------------------------')
#LGBM
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()

start_lgbm = time.time()


if 1 == 1 and 0 == 0:


    print('---------------------------------------------------------------------------------')
    print('Training LGBM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training LGBM', file = f)
    print('---------------------------------------------------------------------------------')
    start = time.time()
    lgbm.fit(X_train_01, y_train_01)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(lgbm, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    joblib.dump(lgbm, 'lgbm_01.joblib')

if 1 == 1:
    lgbm = joblib.load('lgbm_01.joblib')


if 1 == 1:

    print('Prediction LGBM')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction LGBM', file = f)
    print('---------------------------------------------------------------------------------')
    #LGBM
    start = time.time()
    preds_lgbm_01 = lgbm.predict(X_test_01)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

    pred_label = preds_lgbm_01
    name='lgbm'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start_lgbm
    globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### MLP

# %%

#MLP
print('---------------------------------------------------------------------------------')
print('Defining MLP Model')
print('---------------------------------------------------------------------------------')
start_mlp = time.time()


from sklearn.neural_network import MLPClassifier
import time

# create MLPClassifier instance
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)

if 1 == 1 and 0 == 0:


    print('---------------------------------------------------------------------------------')
    print('Training MLP')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
    with open(output_file_name, "a") as f: print('Training MLP', file = f)
    print('---------------------------------------------------------------------------------')

    start = time.time()
    MLP = mlp.fit(X_train_01, y_train_01)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(MLP, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)
    joblib.dump(MLP, 'mlp_01.joblib')

if 1 == 1:
    MLP = joblib.load('mlp_01.joblib')


if 1 == 1:

    #MLP
    start = time.time()
    y_pred = MLP.predict_proba(X_test_01)
    preds_mlp_01 = np.argmax(y_pred,axis = 1)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

#MLP
if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('MLP 01 model', file = f)
    pred_label = preds_mlp_01
    name='mlp'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start_mlp
    globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### ADA

# %%
print('---------------------------------------------------------------------------------')
print('Defining ADA Model')
print('---------------------------------------------------------------------------------')
#ADA
# from sklearn.multioutput import MultiOutputClassifier
start_ada = time.time()


from sklearn.ensemble import AdaBoostClassifier
import time
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

if 1 == 1 and 0 == 0:

    print('---------------------------------------------------------------------------------')
    print('Training ADA')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training ADA', file = f)
    print('---------------------------------------------------------------------------------')
    #ADA


    start = time.time()
    ada = abc.fit(X_train_01, y_train_01)
    end = time.time()

    # # Create the StratifiedKFold object
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # # Perform cross-validation
    # cv_scores = cross_val_score(ada, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
    # # Print the cross-validation scores
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", cv_scores.mean())
    # with open(output_file_name, "a") as f: print('mean accuracy', cv_scores.mean() , file = f)

    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed training time ', time_taken, file = f)

    # Assuming 'model' is your trained model
    joblib.dump(ada, 'ada_01.joblib')




if 1 == 1:
    ada = joblib.load('ada_01.joblib')


if 1 == 1:

    print('Prediction ADA')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Prediction ADA', file = f)
    print('---------------------------------------------------------------------------------')
    #ADA
    start = time.time()
    preds_ada_01 = ada.predict(X_test_01)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    print('---------------------------------------------------------------------------------')

if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('ADA 01 model', file = f)


    pred_label = preds_ada_01
    name='ada'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start_ada
    globals()[f"{name}_time_01"] = time_taken








# %% [markdown]
# ### KNN

# %%
#KNN
print('---------------------------------------------------------------------------------')
print('Defining KNN Model')
print('---------------------------------------------------------------------------------')
start_knn = time.time()

from sklearn.neighbors import KNeighborsClassifier
knn_clf_01=KNeighborsClassifier(n_neighbors = 5)

if 1 == 1 and 0 == 0:

    #KNN
    print('---------------------------------------------------------------------------------')
    print('Training KNN')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training KNN', file = f)
    print('---------------------------------------------------------------------------------')
    start = time.time()
    knn_clf_01.fit(X_train_01,y_train_01)
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
    joblib.dump(knn_clf_01, 'knn_01.joblib')


if load_model_knn == 1:
    knn_clf_01 = joblib.load('knn_01.joblib')

if use_model_knn == 1:

    #KNN
    start = time.time()
    preds_knn =knn_clf_01.predict(X_test_01)
    preds_knn
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('KNN 01 model', file = f)

    pred_label = preds_knn
    name='knn'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC

    end = time.time()
    time_taken = end - start_knn
    globals()[f"{name}_time_01"] = time_taken



# %% [markdown]
# ### Log Regression

# %%
from sklearn.linear_model import LogisticRegression

#Logistic Regression
print('---------------------------------------------------------------------------------')
print('Defining Logistic Regression Model')
print('---------------------------------------------------------------------------------')
logreg_01 = LogisticRegression()
start_lr = time.time()

if 1 == 1 and 0 == 0:

    #KNN
    print('---------------------------------------------------------------------------------')
    print('Training LR ')
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('Training LR', file = f)
    print('---------------------------------------------------------------------------------')
    start = time.time()
    logreg_01.fit(X_train_01,y_train_01)
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
    joblib.dump(logreg_01, 'logreg_01.joblib')


if 1 == 1:
    logreg_01 = joblib.load('logreg_01.joblib')

if 1 == 1:

    #lR
    start = time.time()
    preds_logreg =logreg_01.predict(X_test_01)
    end = time.time()
    time_taken = end - start
    with open(output_file_name, "a") as f: print('Elapsed prediction time ', time_taken, file = f)
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

#LR
if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('LR 01 model', file = f)

    pred_label = preds_logreg
    # pred_label = label[ypred]
    name='lr'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start_lr
    globals()[f"{name}_time_01"] = time_taken


# %%


# %% [markdown]
# ### Catboost

# %%
import catboost
start = time.time()

cat_01 = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')

# Fit the model
cat_01.fit(X_train_01, y_train_01, eval_set=(X_test_01, y_test_01), verbose=10)

# Make predictions on the test set
preds_cat = cat_01.predict(X_test_01)
preds_cat = np.squeeze(preds_cat)

with open(output_file_name, "a") as f: print('--------------------------------------------------------------------------', file = f)

with open(output_file_name, "a") as f: print('catboost', file = f)


pred_label = preds_cat
name='cat'
metrics = confusion_metrics(name, pred_label, y_test_01)

Acc = metrics[0]
Precision = metrics[1]
Recall = metrics[2]
F1 = metrics[3]
BACC = metrics[4]
MCC = metrics[5]    


globals()[f"{name}_acc_01"] = Acc
globals()[f"{name}_pre_01"] = Precision
globals()[f"{name}_rec_01"] = Recall
globals()[f"{name}_f1_01"] = F1
globals()[f"{name}_bacc_01"] = BACC
globals()[f"{name}_mcc_01"] = MCC
end = time.time()
time_taken = end - start
globals()[f"{name}_time_01"] = time_taken


# %% [markdown]
# ### XGB

# %%

import xgboost as xgb
start = time.time()

# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_01, label=y_train_01)
dtest = xgb.DMatrix(X_test_01, label=y_test_01)

# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': 5,  # specify the number of classes
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'  # metric for multi-class classification
}

# Train the XGBoost model
num_round = 100
xgb_01 = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
preds_xgb_01 = xgb_01.predict(dtest)


if 1 == 1:

    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f: print('xgboost base model', file = f)

    pred_label = preds_xgb_01
    name='xgb'
    metrics = confusion_metrics(name, pred_label, y_test_01)

    Acc = metrics[0]
    Precision = metrics[1]
    Recall = metrics[2]
    F1 = metrics[3]
    BACC = metrics[4]
    MCC = metrics[5]    


    globals()[f"{name}_acc_01"] = Acc
    globals()[f"{name}_pre_01"] = Precision
    globals()[f"{name}_rec_01"] = Recall
    globals()[f"{name}_f1_01"] = F1
    globals()[f"{name}_bacc_01"] = BACC
    globals()[f"{name}_mcc_01"] = MCC
    end = time.time()
    time_taken = end - start
    globals()[f"{name}_time_01"] = time_taken


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
#  ### Generating Summary Metric Table

# %%
from tabulate import tabulate

# Assuming data is a 110x4 list, where each row is a sublist
# data =  [["Row {} Col {}".format(i + 1, j + 1) for j in range(4)] for i in range(110)]
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
                'VOTING',
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
                'avg',
                'weighed_avg'
                ]

data = [["" for _ in range(5)] for _ in range(len(names_models))]

level_01_acc = [
                ada_acc_01,
                svm_acc_01,
                dnn_acc_01,
                mlp_acc_01,
                knn_acc_01,
                cat_acc_01,
                xgb_acc_01,
                lgbm_acc_01,
                rf_acc_01,
                lr_acc_01,
                voting_acc_01,
                bag_svm_acc_01,
                bag_knn_acc_01,
                bag_dt_acc_01,
                bag_lr_acc_01,
                bag_mlp_acc_01,

                bag_rf_acc_01,
                bag_ada_acc_01,
                bag_lgbm_acc_01,
                # bag_xgb_acc_01,
                bag_cat_acc_01,
                bag_comb_acc_01,

                avg_acc_01,
                weighed_avg_acc_01
                ]  


level_01_pre = [
                ada_pre_01,
                svm_pre_01,
                dnn_pre_01,
                mlp_pre_01,
                knn_pre_01,
                cat_pre_01,
                xgb_pre_01,
                lgbm_pre_01,
                rf_pre_01,
                lr_pre_01,
                voting_pre_01,
                bag_svm_pre_01,
                bag_knn_pre_01,
                bag_dt_pre_01,
                bag_lr_pre_01,
                bag_mlp_pre_01,

                bag_rf_pre_01,
                bag_ada_pre_01,
                bag_lgbm_pre_01,
                # bag_xgb_pre_01,
                bag_cat_pre_01,
                bag_comb_pre_01,

                avg_pre_01,
                weighed_avg_pre_01
                ]  

level_01_rec = [
                ada_rec_01,
                svm_rec_01,
                dnn_rec_01,
                mlp_rec_01,
                knn_rec_01,
                cat_rec_01,
                xgb_rec_01,
                lgbm_rec_01,
                rf_rec_01,
                lr_rec_01,
                voting_rec_01,
                bag_svm_rec_01,
                bag_knn_rec_01,
                bag_dt_rec_01,
                bag_lr_rec_01,
                bag_mlp_rec_01,

                bag_rf_rec_01,
                bag_ada_rec_01,
                bag_lgbm_rec_01,
                # bag_xgb_rec_01,
                bag_cat_rec_01,
                bag_comb_rec_01,

                avg_rec_01,
                weighed_avg_rec_01
                ]  

level_01_f1 = [
                ada_f1_01,
                svm_f1_01,
                dnn_f1_01,
                mlp_f1_01,
                knn_f1_01,
                cat_f1_01,
                xgb_f1_01,
                lgbm_f1_01,
                rf_f1_01,
                lr_f1_01,
                voting_f1_01,
                bag_svm_f1_01,
                bag_knn_f1_01,
                bag_dt_f1_01,
                bag_lr_f1_01,
                bag_mlp_f1_01,

                bag_rf_f1_01,
                bag_ada_f1_01,
                bag_lgbm_f1_01,
                # bag_xgb_f1_01,
                bag_cat_f1_01,
                bag_comb_f1_01,

                avg_f1_01,
                weighed_avg_f1_01
                ]  




# Combine data into a list of tuples for sorting
model_data = list(zip(names_models, level_01_acc, level_01_pre, level_01_rec, level_01_f1))

# Sort by F1-01 score in descending order
model_data_sorted = sorted(model_data, key=lambda x: x[4], reverse=True)

# Separate the sorted data back into individual lists
sorted_names_models, sorted_level_01_acc, sorted_level_01_pre, sorted_level_01_rec, sorted_level_01_f1 = zip(*model_data_sorted)

# Assign the sorted data to the table
for i in range(len(sorted_names_models)):
    data[i][0] = sorted_names_models[i]
    data[i][1] = sorted_level_01_acc[i]
    data[i][2] = sorted_level_01_pre[i] 
    data[i][3] = sorted_level_01_rec[i] 
    data[i][4] = sorted_level_01_f1[i]

# Define column headers
headers = ["Models", "ACC-01", "PRE-01", "REC-01", "F1-01"]

# Print the table
table = tabulate(data, headers=headers, tablefmt="grid")
with open(output_file_name, "a") as f: print('Summary table', file = f)
if pick_prob == 1: 
    with open(output_file_name, "a") as f: print('Level 01 - Probabilities', file = f)
else:
    with open(output_file_name, "a") as f: print('Level 01 - CLASSES', file = f)
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
                'VOTING',
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
                'avg',
                'weighed_avg'
                ]

data = [["" for _ in range(2)] for _ in range(len(names_models))]

level_01_time = [
                ada_time_01,
                svm_time_01,
                dnn_time_01,
                mlp_time_01,
                knn_time_01,
                cat_time_01,
                xgb_time_01,
                lgbm_time_01,
                rf_time_01,
                lr_time_01,
                voting_time_01,
                bag_svm_time_01,
                bag_knn_time_01,
                bag_dt_time_01,
                bag_lr_time_01,
                bag_mlp_time_01,

                bag_rf_time_01,
                bag_ada_time_01,
                bag_lgbm_time_01,
                # bag_xgb_time_01,
                bag_cat_time_01,
                bag_comb_time_01,

                avg_time_01,
                weighed_avg_time_01
                ]  


# Combine data into a list of tuples for sorting
model_data = list(zip(names_models, level_01_time))

# Sort by F1-01 score in descending order
model_data_sorted = sorted(model_data, key=lambda x: x[1], reverse=False)

# Separate the sorted data back into individual lists
sorted_names_models, sorted_level_01_time = zip(*model_data_sorted)

# Assign the sorted data to the table
for i in range(len(sorted_names_models)):
    data[i][0] = sorted_names_models[i]
    data[i][1] = sorted_level_01_time[i]

# Define column headers
headers = ["Models", "time-01(sec)"]


# Print the table
table = tabulate(data, headers=headers, tablefmt="grid")
with open(output_file_name, "a") as f: print('Time is counted is seconds', file = f)
print(table)
with open(output_file_name, "a") as f: print(table, file = f)
end_program = time.time()
time_program = end_program - start_program
with open(output_file_name, "a") as f: print('Running time of entire program is:', time_program ,' seconds',file = f)

# %% [markdown]
# # ------------------------------------------------------------------

# %%


# %% [markdown]
# ### Feature Selection

# %%
if generate_feature_importance == 1:
  print('---------------------------------------------------------------------------------')
  print('Generating SHAP explanation')
  print('---------------------------------------------------------------------------------')
  print('')
  with open(output_file_name, "a") as f:print('ADA FEATURE IMPORTANCE',file = f)

      #START TIMER MODEL
  start = time.time()

  print('---------------------------------------------------------------------------------')
  print('Generating explainer')
  print('---------------------------------------------------------------------------------')
  print('')
  test = X_test_01
  train = X_train_01
  # ## Summary Bar Plot Global
  start_index = 0
  end_index = 250
  # test.pop('Label')
  # test.pop('is_train')
  # print(label2)


  # models = [ada,dnn_01,clf,knn_clf_01,cat_01,xgb_01, rf, lgbm, mlp,logreg_01]
  explainer = shap.KernelExplainer(ada.predict_proba, test[start_index:end_index])

  shap_values = explainer.shap_values(test[start_index:end_index])

  shap.summary_plot(shap_values = shap_values,
                    features = test[start_index:end_index],
                    # class_names=[column_features[:-1]],
                    show=False)

  # if feature_selection_bit == 1 # On
  # pick_prob = 0 # set equal one to choose the dataset with probabilities, set to 0 to choose one with the classes.
  if pick_prob == 1:
    plt.savefig('ADA_SHAP_NSL_prob_01.png')
  elif pick_prob == 0:
    plt.savefig('ADA_SHAP_NSL_class_01.png')
        
  else: None
  plt.clf()


  vals= np.abs(shap_values).mean(1)
  feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
  feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
  feature_importance.head()
  print(feature_importance.to_string())

  with open(output_file_name, "a") as f:print('Feature Importance: ',feature_importance.to_string(),file = f)



  end = time.time()
  with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min',file = f)
  print('---------------------------------------------------------------------------------')



  print('---------------------------------------------------------------------------------')
  # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
  feature_val = feature_importance['feature_importance_vals'].tolist()

  # col_name = 'col_name'  # Replace with the name of the column you want to extract
  feature_name = feature_importance['col_name'].tolist()


  # for item1, item2 in zip(feature_name, feature_val):
  #     print(item1, item2)


  # Use zip to combine the two lists, sort based on list1, and then unzip them
  zipped_lists = list(zip(feature_name, feature_val))
  zipped_lists.sort(key=lambda x: x[1],reverse=True)

  # Convert the sorted result back into separate lists
  sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='', file = f)

  # with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
  # with open(output_file_name, "a") as f:print("]", file = f)

  print('---------------------------------------------------------------------------------')




# %%

# explainer = shap.TreeExplainer(model)
# start_index = 0
# end_index = samples
# shap_values = explainer.shap_values(test[start_index:end_index])
# shap_obj = explainer(test[start_index:end_index])
# shap.summary_plot(shap_values = shap_values,
#                   features = test[start_index:end_index],
#                 show=False)
# plt.savefig('Light_SHAP_CIC_Summary.png')
# plt.clf()


# vals= np.abs(shap_values).mean(1)
# feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
# feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
# feature_importance.head()
# print(feature_importance.to_string())
# print('---------------------------------------------------------------------------------')
# # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
# feature_val = feature_importance['feature_importance_vals'].tolist()

# # col_name = 'col_name'  # Replace with the name of the column you want to extract
# feature_name = feature_importance['col_name'].tolist()


# # for item1, item2 in zip(feature_name, feature_val):
# #     print(item1, item2)


# # Use zip to combine the two lists, sort based on list1, and then unzip them
# zipped_lists = list(zip(feature_name, feature_val))
# zipped_lists.sort(key=lambda x: x[1],reverse=True)

# # Convert the sorted result back into separate lists
# sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# for k in sorted_list1:
#   with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)

# # with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
# for k in sorted_list1:
#   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
# with open(output_file_name, "a") as f:print("]", file = f)
# print('---------------------------------------------------------------------------------')


# %%
if generate_feature_importance == 1:

  print('---------------------------------------------------------------------------------')
  print('Generating SHAP explanation')
  print('---------------------------------------------------------------------------------')
  print('')

  with open(output_file_name, "a") as f:print('XGB FEATURE IMPORTANCE',file = f)

      #START TIMER MODEL
  start = time.time()

  print('---------------------------------------------------------------------------------')
  print('Generating explainer')
  print('---------------------------------------------------------------------------------')
  print('')
  test = X_test_01
  train = X_train_01
  # ## Summary Bar Plot Global
  start_index = 0
  end_index = 250
  # test.pop('Label')
  # test.pop('is_train')
  # print(label2)


  explainer = shap.TreeExplainer(xgb_01)

  shap_values = explainer.shap_values(test[start_index:end_index])
  shap_obj = explainer(test[start_index:end_index])
  shap.summary_plot(shap_values = shap_values,
                    features = test[start_index:end_index],
                  show=False)
  # plt.clf()

  # if feature_selection_bit == 1 # On
  # pick_prob = 0 # set equal one to choose the dataset with probabilities, set to 0 to choose one with the classes.
  if pick_prob == 1:
    plt.savefig('XGB_SHAP_NSL_prob_01.png')
  elif pick_prob == 0:
    plt.savefig('XGB_SHAP_NSL_class_01.png')

  else: None
  plt.clf()


  vals= np.abs(shap_values).mean(1)
  feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
  feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
  feature_importance.head()
  print(feature_importance.to_string())
  with open(output_file_name, "a") as f:print('Feature Importance: ',feature_importance.to_string(),file = f)



  end = time.time()
  with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min',file = f)
  print('---------------------------------------------------------------------------------')



  print('---------------------------------------------------------------------------------')
  # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
  feature_val = feature_importance['feature_importance_vals'].tolist()

  # col_name = 'col_name'  # Replace with the name of the column you want to extract
  feature_name = feature_importance['col_name'].tolist()


  # for item1, item2 in zip(feature_name, feature_val):
  #     print(item1, item2)


  # Use zip to combine the two lists, sort based on list1, and then unzip them
  zipped_lists = list(zip(feature_name, feature_val))
  zipped_lists.sort(key=lambda x: x[1],reverse=True)

  # Convert the sorted result back into separate lists
  sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='', file = f)

  # with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
  # with open(output_file_name, "a") as f:print("]", file = f)

  print('---------------------------------------------------------------------------------')




# %%
if generate_feature_importance == 1:


  print('---------------------------------------------------------------------------------')
  print('Generating SHAP explanation')
  print('---------------------------------------------------------------------------------')
  print('')

  with open(output_file_name, "a") as f:print('LGBM FEATURE IMPORTANCE',file = f)

      #START TIMER MODEL
  start = time.time()

  print('---------------------------------------------------------------------------------')
  print('Generating explainer')
  print('---------------------------------------------------------------------------------')
  print('')
  test = X_test_01
  train = X_train_01
  # ## Summary Bar Plot Global
  start_index = 0
  end_index = 250
  # test.pop('Label')
  # test.pop('is_train')
  # print(label2)


  explainer = shap.TreeExplainer(lgbm)

  shap_values = explainer.shap_values(test[start_index:end_index])
  shap_obj = explainer(test[start_index:end_index])
  shap.summary_plot(shap_values = shap_values,
                    features = test[start_index:end_index],
                  show=False)
  # plt.clf()

  # if feature_selection_bit == 1 # On
  # pick_prob = 0 # set equal one to choose the dataset with probabilities, set to 0 to choose one with the classes.
  if pick_prob == 1:
    plt.savefig('LGBM_SHAP_NSL_prob_01.png')
  elif pick_prob == 0:
    plt.savefig('LGBM_SHAP_NSL_class_01.png')

  else: None
  plt.clf()


  vals= np.abs(shap_values).mean(1)
  feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
  feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
  feature_importance.head()
  print(feature_importance.to_string())
  with open(output_file_name, "a") as f:print('Feature Importance: ',feature_importance.to_string(),file = f)



  end = time.time()
  with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min',file = f)
  print('---------------------------------------------------------------------------------')



  print('---------------------------------------------------------------------------------')
  # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
  feature_val = feature_importance['feature_importance_vals'].tolist()

  # col_name = 'col_name'  # Replace with the name of the column you want to extract
  feature_name = feature_importance['col_name'].tolist()


  # for item1, item2 in zip(feature_name, feature_val):
  #     print(item1, item2)


  # Use zip to combine the two lists, sort based on list1, and then unzip them
  zipped_lists = list(zip(feature_name, feature_val))
  zipped_lists.sort(key=lambda x: x[1],reverse=True)

  # Convert the sorted result back into separate lists
  sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='', file = f)

  # with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
  # with open(output_file_name, "a") as f:print("]", file = f)

  print('---------------------------------------------------------------------------------')




# %%
if generate_feature_importance == 1:

  print('---------------------------------------------------------------------------------')
  print('Generating SHAP explanation')
  print('---------------------------------------------------------------------------------')
  print('')

  with open(output_file_name, "a") as f:print('RF FEATURE IMPORTANCE',file = f)

      #START TIMER MODEL
  start = time.time()

  print('---------------------------------------------------------------------------------')
  print('Generating explainer')
  print('---------------------------------------------------------------------------------')
  print('')
  test = X_test_01
  train = X_train_01
  # ## Summary Bar Plot Global
  start_index = 0
  end_index = 250
  # test.pop('Label')
  # test.pop('is_train')
  # print(label2)


  explainer = shap.TreeExplainer(rf)

  shap_values = explainer.shap_values(test[start_index:end_index])
  shap_obj = explainer(test[start_index:end_index])
  shap.summary_plot(shap_values = shap_values,
                    features = test[start_index:end_index],
                  show=False)
  # plt.clf()

  # if feature_selection_bit == 1 # On
  # pick_prob = 0 # set equal one to choose the dataset with probabilities, set to 0 to choose one with the classes.
  if pick_prob == 1:
    plt.savefig('RF_SHAP_NSL_prob_01.png')
  elif pick_prob == 0:
    plt.savefig('RF_SHAP_NSL_class_01.png')

  else: None
  plt.clf()


  vals= np.abs(shap_values).mean(1)
  feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
  feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
  feature_importance.head()
  print(feature_importance.to_string())
  with open(output_file_name, "a") as f:print('Feature Importance: ',feature_importance.to_string(),file = f)



  end = time.time()
  with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min',file = f)
  print('---------------------------------------------------------------------------------')



  print('---------------------------------------------------------------------------------')
  # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
  feature_val = feature_importance['feature_importance_vals'].tolist()

  # col_name = 'col_name'  # Replace with the name of the column you want to extract
  feature_name = feature_importance['col_name'].tolist()


  # for item1, item2 in zip(feature_name, feature_val):
  #     print(item1, item2)


  # Use zip to combine the two lists, sort based on list1, and then unzip them
  zipped_lists = list(zip(feature_name, feature_val))
  zipped_lists.sort(key=lambda x: x[1],reverse=True)

  # Convert the sorted result back into separate lists
  sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='', file = f)

  # with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
  # for k in sorted_list1:
  #   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
  # with open(output_file_name, "a") as f:print("]", file = f)

  print('---------------------------------------------------------------------------------')





