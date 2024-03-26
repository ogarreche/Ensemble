# A Two-Level Ensemble Learning Framework for Enhancing Network Intrusion Detection Systems

### Abstract

The exponential growth of intrusions on networked systems inspires new research directions on developing artificial intelligence (AI) techniques for intrusion detection systems (IDS). In this context, several AI techniques have been leveraged for automating network intrusion detection tasks. However, each AI model has unique strengths points and weaknesses, and one may be better than the other depending on the dataset,
which might aggravate which model to choose. Thus, combining these AI models can improve their use of generalization and application in network
intrusion detection tasks. In this paper, we aim to fill such a gap by evaluating diverse ensemble methods for network intrusion detection systems. In particular, we build a two-level ensemble learning framework for
evaluating such ensemble learning methods in network
intrusion detection tasks. In the first level of our framework, we load the input dataset, train the base learners and ensemble methods, and generate the evaluation metrics. This level also produces new datasets (needed to train the second level) based on both prediction probabilities of base and
ensemble models used in the first level. The second level of the framework consists of loading the datasets
generated from the first level, training the ensemble methods, and generating the evaluation metrics. Our framework also considers feature selection for both levels. In particular, we perform XAI-based feature selection in the first level and Information Gain-based feature selection in the second level.  We present results for several ensemble model combinations in our two-level framework (i.e., 24 methods), including different bagging, stacking, and boosting methods on several base learners (e.g., decision trees, support vector machines, deep neural networks, and others). We evaluate our framework on three network intrusion
datasets with different characteristics (RoEduNet-SIMARGL2021, NSL-KDD, and CICIDS-2017). We also categorize AI models according to their performances on our evaluation metrics.  
Our evaluation shows that it is beneficial to perform
two-level learning for most setups considered in this work. We also release our source codes for the community to access as a baseline two-level ensemble learning framework for network intrusion detection.

Figure 1 - High Level Framework
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/framework.png?raw=true)

Figure 2 - Low Level Framework
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/low_level_framework.png?raw=true)

Figure 3 - Summary of main results.
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/Summary.png?raw=true)



### Datasets:

Download one of the datasets. 

RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017

NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

### How to use the programs:

Inside the CICIDS or SIMARGL or NSLKDD folder you will find programs for each model used in this paper. Each one of these programs outputs:

  - Level 00 and Level 01 programs outputs: The Accuracy, Recall, Precision, F1 and time efficiency for every AI model. (Run Level 00 before Level01)
  - The FPR values are generated "manually". After running LV00 or LV01 program, a text file with the results with a confusion matrix will be generated for each model. Copy and paste each decision matrix in the FPR.ipynb program to generate the FPR results.
  - Run the wilcoxon programs to generate the statistics restults (These programs can be run independtly of Level00 and Level01).  
 
    
### Status:
  - Paper under IEEE Access Revision.
