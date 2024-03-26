# %%
import pandas as pd


req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
           'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
           'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
            'ALERT']

req_cols =  [ 
    'TCP_WIN_SCALE_IN', 
    'TCP_WIN_MIN_IN', 
    'TCP_WIN_MAX_IN', 
    'TCP_WIN_MSS_IN', 
    'TCP_FLAGS',
    'PROTOCOL', 
    'FLOW_DURATION_MILLISECONDS', 
    # 'TCP_WIN_MAX_OUT', 
    'TCP_WIN_MIN_OUT', 
    # 'SRC_TOS', 
    # 'DST_TOS',
    'ALERT' 
    ]

address = '/home/oarreche@ads.iu.edu/HITL/sensor/sensor_db'
print('Loading Database')
print('--------------------------------------------------')

fraction = 1
#Denial of Service
df0 = pd.read_csv (address + '/dos-03-15-2022-15-44-32.csv', usecols=req_cols).sample(frac = fraction)
df1 = pd.read_csv (address + '/dos-03-16-2022-13-45-18.csv', usecols=req_cols).sample(frac = fraction)
df2 = pd.read_csv (address + '/dos-03-17-2022-16-22-53.csv', usecols=req_cols).sample(frac = fraction)
df3 = pd.read_csv (address + '/dos-03-18-2022-19-27-05.csv', usecols=req_cols).sample(frac = fraction)
df4 = pd.read_csv (address + '/dos-03-19-2022-20-01-53.csv', usecols=req_cols).sample(frac = fraction)
df5 = pd.read_csv (address + '/dos-03-20-2022-14-27-54.csv', usecols=req_cols).sample(frac = fraction)


#Malware
#df6 = pd.read_csv ('sensor_db/malware-03-25-2022-17-57-07.csv', usecols=req_cols)

#Normal
df7 = pd.read_csv  (address + '/normal-03-15-2022-15-43-44.csv', usecols=req_cols).sample(frac = fraction)
df8 = pd.read_csv  (address + '/normal-03-16-2022-13-44-27.csv', usecols=req_cols).sample(frac = fraction)
df9 = pd.read_csv  (address + '/normal-03-17-2022-16-21-30.csv', usecols=req_cols).sample(frac = fraction)
df10 = pd.read_csv (address + '/normal-03-18-2022-19-17-31.csv', usecols=req_cols).sample(frac = fraction)
df11 = pd.read_csv (address + '/normal-03-18-2022-19-25-48.csv', usecols=req_cols).sample(frac = fraction)
df12 = pd.read_csv (address + '/normal-03-19-2022-20-01-16.csv', usecols=req_cols).sample(frac = fraction)
df13 = pd.read_csv (address + '/normal-03-20-2022-14-27-30.csv', usecols=req_cols).sample(frac = fraction)


#PortScanning
df14 = pd.read_csv  (address + '/portscanning-03-15-2022-15-44-06.csv', usecols=req_cols).sample(frac = fraction)
df15 = pd.read_csv  (address + '/portscanning-03-16-2022-13-44-50.csv', usecols=req_cols).sample(frac = fraction)
df16 = pd.read_csv  (address + '/portscanning-03-17-2022-16-22-53.csv', usecols=req_cols).sample(frac = fraction)
df17 = pd.read_csv  (address + '/portscanning-03-18-2022-19-27-05.csv', usecols=req_cols).sample(frac = fraction)
df18 = pd.read_csv  (address + '/portscanning-03-19-2022-20-01-45.csv', usecols=req_cols).sample(frac = fraction)
df19 = pd.read_csv  (address + '/portscanning-03-20-2022-14-27-49.csv', usecols=req_cols).sample(frac = fraction)



frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
df = pd.concat(frames,ignore_index=True)

# shuffle the DataFrame rows
df = df.sample(frac =0.005)

# assign alert column to y
y = df.pop('ALERT')

# join alert back to df
df = df.assign( ALERT = y) 

#Fill NaN with 0s
df = df.fillna(0)


# %%
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn
split = 0.8
filtered_normal = df[df['ALERT'] == 'None']

#reduce

reduced_normal = filtered_normal.sample(frac=0.2)

#join

df = pd.concat([df[df['ALERT'] != 'None'], reduced_normal])


# Normalize database
print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

# make a df copy
df_max_scaled = df.copy()

# assign alert column to y
y = df_max_scaled.pop('ALERT')

#Normalize operation
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
#assign df copy to df
df = df_max_scaled.assign( ALERT = y)
#Fill NaN with 0s
df = df.fillna(0)

y = df.pop('ALERT')
X = df

y, y_Label = pd.factorize(y)

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split,random_state=42)
df = X.assign( ALERT = y)


# %%


# %%
#Model Parameters RF
from sklearn.ensemble import RandomForestClassifier
max_depth = 5
n_estimators = 5
min_samples_split = 2

print('---------------------------------------------------------------------------------')
print('Defining the RF model')
print('---------------------------------------------------------------------------------')
print('')

rf = RandomForestClassifier(max_depth = max_depth,  n_estimators = n_estimators, min_samples_split = min_samples_split, n_jobs = -1)

# model = clf.fit(X, y)

# %%
#Model Parameters SVM
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

max_iter=10
loss='hinge'
gamma=0.1


rbf_feature = RBFSampler(gamma=gamma, random_state=1)
X_features = rbf_feature.fit_transform(X_train)
svm = SGDClassifier(max_iter=max_iter,loss=loss)
# clf.fit(X_features, y_train)
# clf.score(X_features, y_train)


# %%
#Model Parameters ADA
from sklearn.ensemble import AdaBoostClassifier
n_estimators=100
learning_rate=0.5

ada = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)


# %%

# #Model Parameters DNN
# import tensorflow as tf
# dropout_rate = 0.01
# nodes = 70
# out_layer = 7
# optimizer='adam'
# loss='sparse_categorical_crossentropy'
# epochs=5
# batch_size=2*256


# print('---------------------------------------------------------------------------------')
# print('Defining the DNN model')
# print('---------------------------------------------------------------------------------')
# print('')


# num_columns = X_train.shape[1]

# dnn = tf.keras.Sequential()

# # Input layer
# dnn.add(tf.keras.Input(shape=(num_columns,)))

# # Dense layers with dropout
# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# # Output layer
# dnn.add(tf.keras.layers.Dense(out_layer))



# dnn.compile(optimizer=optimizer, loss=loss)

# dnn.summary()

import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor

def getModel(optimizer):


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
    dnn.add(tf.keras.layers.Dense(out_layer))



    dnn.compile(optimizer=optimizer, loss=loss)

    return dnn


#Model Parameters DNN
dropout_rate = 0.01
nodes = 70
out_layer = 7
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=5
batch_size=2*256
# optimizer = ['Adam']
epochs = [5]


param_grid = dict(epochs=epochs, optimizer=optimizer)
dnn_Kmodel = KerasClassifier(build_fn=getModel, verbose=1)

# %%
#Model Parameters KNN
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=5

knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)

# %%
#Model Parameters LGBM
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
n_split= 3
n_repeat= 15

lgbm = LGBMClassifier()
# cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat)
# n_scores = cross_val_score(lgbm, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# fit the model on the whole dataset
# lgbm = LGBMClassifier()



# %%
#Model Parameters MLP
from sklearn.neural_network import MLPClassifier
max_iter=70
# MLP = MLPClassifier(random_state=1, max_iter=max_iter).fit(X_train, y_train)
MLP = MLPClassifier(random_state=42, max_iter=max_iter)


# %%
from sklearn.tree import DecisionTreeClassifier
# Create a Decision Tree Classifier
DT = DecisionTreeClassifier(random_state=42)

from sklearn.ensemble import BaggingClassifier
# # Define the base classifier (Decision Tree in this case)
base_classifier = DT

# Define the BaggingClassifier
bag_DT = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)


from sklearn.linear_model import LogisticRegression
#Logistic Regression
LR = LogisticRegression()
#catboost
import catboost
CAT = catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', custom_metric='Accuracy')


# import xgboost as xgb
# # Create a DMatrix for XGBoost
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
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
# XGB = xgb.train(params, dtrain, num_round)
# # Make predictions on the test set
# preds_xgb = XGB.predict(dtest)
# # preds_xgb_prob = xgb_00.predict_proba(dtest)


# Get class probabilities
# Assuming binary classification, get the probability for the positive class (class 1)
# preds_xgb_margin = XGB.predict(dtest, output_margin=True)
# preds_xgb_prob = 1 / (1 + np.exp(-preds_xgb_margin))

# %%
output_file_name = "SML_WIL_final.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)




# %%

# %%
from scipy.stats import wilcoxon
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Prepare models and select your CV method
# model1 = ExtraTreesClassifier()
# model2 = RandomForestClassifier()
model1 = svm
model2 = rf
model3 = ada
model4 = dnn_Kmodel
model5 = lgbm
model6 = MLP
model7 = knn_clf
model8 = LR
model9 = DT
model10 = CAT
# model11 = XGB

model12 = bag_DT
# model13 = bag_RF
# model14 = bag_ADA
# model15 = bag_CAT
# model16 = bag_LGBM
# model17 = bag_KNN
# model18 = bag_COMB
# model19 = bag_MLP
# model20 = bag_SVM
# model21 = bag_LR           

name_models = [
                'SVM',
                'RF',
                'DNN',
                'LGBM',
                'MLP',
                'KNN',
                'LR',
                'DT',
                'CAT',
                # 'XGB',
                'bag_DT',
                'ADA'
                # 'bag_RF',
                # 'bag_ADA',
                # 'bag_CAT',
                # 'bag_LGBM',
                # 'bag_KNN',
                # 'bag_COMB',
                # 'bag_MLP',
                # 'bag_SVM',
                # 'bag_LR'            
               ]

actual_models = [
                svm,
                rf,
                dnn_Kmodel,
                lgbm,
                MLP,
                knn_clf,
                LR,
                DT,
                CAT,
                # XGB
                bag_DT,
                ada

]
kf = KFold(n_splits=10, random_state=None)

y , yL = pd.factorize(y)

# %%
import numpy as np
def crosstrain(model):
    results_model = cross_val_score(model, X, y, cv=kf) 
    return results_model

def wilcox(model1_str,results_model1,model2_str,results_model2):
    # results_model1 = cross_val_score(model_1, X, y, cv=kf) #SVM
    # results_model2 = cross_val_score(model_2, X, y, cv=kf) # RF

    # %%
    # Calculate p value
    #SVM and RF
    with open(output_file_name, "a") as f:print(model1_str +" and " + model2_str,file = f)
    stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit'); p
    median_model1 = np.median(results_model1)
    median_model2 = np.median(results_model2)
    with open(output_file_name, "a") as f:print("p value: ", p , file = f)
    if p < 0.05:
        if median_model1 > median_model2:
            print(model1_str + " is better.")
            with open(output_file_name, "a") as f:print(model1_str +" is better.", file = f)
        elif median_model1 < median_model2:
            print(model2_str +" is better.")
            with open(output_file_name, "a") as f:print(model2_str +" is better.", file = f)
        else:
            print("Models are statistically different but have the same median.")
            with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
    else:
        print("No statistically significant difference.")
        with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)
    return None

# %%
# for i in range(0,len(name_models)):
#     for j in range (i,len(name_models)):
#         if i != j:
#             print (name_models [i],' and ',name_models[j])

# %%
# for i in range(0,len(name_models)):
#     for j in range (i,len(name_models)):
#         if i != j:
#             print (name_models [i],' and ',name_models[j])
#             model_name1 = name_models [i]
#             model_name2 = name_models [j]
#             actual_model1 = actual_models [i]
#             actual_model2 = actual_models [j]
#             wilcox(model_name1,actual_model1,model_name2,actual_model2)
results = []
for j in range (0,len(name_models)):
    results.append(crosstrain(actual_models[j]))
    print(name_models[j], "done")

# single_name =  'LGBM'
# single_model = lgbm 
# for j in range (0,len(name_models)):
#     if single_name != name_models[j]:
#         print (single_name,' and ',name_models[j])
#         model_name1 = single_name
#         model_name2 = name_models [j]
#         actual_model1 = single_model
#         actual_model2 = actual_models [j]
#         wilcox(model_name1,actual_model1,model_name2,actual_model2)
    
#xgb as corner case
    

# import xgboost as xgb
# # Create a DMatrix for XGBoost
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
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
# XGB = xgb.train(params, dtrain, num_round)
# # Make predictions on the test set
# preds_xgb = XGB.predict(dtest)
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold

# Define your XGBoost model
model = xgb.XGBClassifier(objective='multi:softmax', max_depth=3, learning_rate=0.1)
# Define your cross-validation strategy
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the parameters for early stopping
early_stopping_rounds = 10  # Stop if the score doesn't improve for 10 consecutive rounds

# Perform cross-validation with early stopping
result_xgb = cross_val_score(model, X, y, cv=kf, 
                             fit_params={"early_stopping_rounds": early_stopping_rounds, 
                                         "eval_metric": "logloss", 
                                         "eval_set": [(X, y)]})
# Split your data into features (X) and labels (y)
# Replace X and y with your actual data


# Define your cross-validation strategy
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
# result_xgb = cross_val_score(model, X, y, cv=kf)
name_models.append('XGB')
results.append(result_xgb)
print('XGB done')



print("print results")
for i in range(0,len(name_models)):
    for j in range (i,len(name_models)):
        if i != j:
            print (name_models [i],' and ',name_models[j])
            model_name1 = name_models [i]
            model_name2 = name_models [j]
            actual_model1 = results [i]
            actual_model2 = results [j]
            wilcox(model_name1,actual_model1,model_name2,actual_model2)


