# %%
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']

req_cols = [ 
                ' Destination Port',
                ' Init_Win_bytes_backward',
                ' Packet Length Std',
                ' Bwd Packet Length Mean', 
                ' Total Length of Bwd Packets', 
                ' Packet Length Mean',
                ' Subflow Bwd Bytes',
                ' Packet Length Variance', 
                ' Label']

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

# frames = [df0, df1, df2, df3, df4, df5,df6, df7]

df = pd.concat(frames,ignore_index=True)
df = df.sample(frac = 0.3)
# df = pd.concat(frames,ignore_index=True)
# df = df.sample(frac = fraction )
y = df.pop(' Label')
df = df.assign(Label = y)


# Specify the name of the output text file
output_file_name = "Wilcoxon_CIC_final.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################
###################################################

# %%
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn
print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')

frac_normal = 0.2
split = 0.8

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


print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

df_max_scaled
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
#df
df = df.fillna(0)

y = df.pop('Label')
X = df
df = df.assign(Label = y)

counter = Counter(y)
print(counter)


X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split,random_state=42)
df = X.assign( Label = y)
# df[['bp_before','bp_after']].describe()

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
# n_split= 3
# n_repeat= 15

lgbm = LGBMClassifier(max_depth = 5, num_leaves = 10, early_stopping_rounds=10)
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
# from sklearn.preprocessing import LabelEncoder

# # Initialize LabelEncoder
# label_encoder = LabelEncoder()

# # Fit and transform the labels
# y = label_encoder.fit_transform(y)
# print(y)
# dtrain = xgb.DMatrix(X, label=y)
# # dtest = xgb.DMatrix(X_test, label=y_test)
# # Set XGBoost parameters
# params = {
#     'objective': 'multi:softmax',  # for multi-class classification
#     'num_class': len(label_encoder.classes_),  # specify the number of classes
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'eval_metric': 'mlogloss'  # metric for multi-class classification
# }

# # Train the XGBoost model
# num_round = 10
# XGB = xgb.train(params, dtrain, num_round)
# Make predictions on the test set
# preds_xgb = XGB.predict(dtest)
# preds_xgb_prob = xgb_00.predict_proba(dtest)


# # Get class probabilities
# # Assuming binary classification, get the probability for the positive class (class 1)
# preds_xgb_margin = XGB.predict(dtest, output_margin=True)
# preds_xgb_prob = 1 / (1 + np.exp(-preds_xgb_margin))




y , yL = pd.factorize(y)


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


