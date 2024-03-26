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

### Framework:

Figure 1 - A high-level overview of our Ensemble Learning framework for network intrusion detection. It considers a diverse set of AI models and network intrusion datasets.
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/framework.png?raw=true)

Figure 1 shows two major areas (i.e., Level 00 and Level
01) divided by a horizontal line, indicated by the blocks on
the left-handed side in a vertical position. The diagram is
read from bottom to top, analogous to a pyramid that starts
by building a strong foundation and then grows vertically
with stacking, and its vertical layout is due to the ensemble
learning methods applied in this work.

Figure 2 - A low-level overview of our Ensemble Learning framework for network intrusion detection. It considers a diverse set of AI models and network intrusion datasets.
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/low_level_framework.png?raw=true)

We now explain the Low-Level Ensemble Learning Pipeline
Components (shown in Figure 2). The reader may follow the
arrows in the block diagram to facilitate its readability. Each
arrow type has a color code: black refers to processes in Level
00, orange relates to processes in Level 01, and the green
arrow marks the transition from Level 00 to Level 01. Each
block group has sub-groups that indicate that they are a set or
included in the context of the outer block. Also, note that some
blocks have different colors for easy identification (e.g., the
base models are gray). These blocks seen before may appear
in another context (i.e., inside other blocks). For example,
note that base models (gray) appear as a small gray block
inside stacking.

### Main Results:

Figure 3 - The summary of all results in this work. Our framework provides higher performance metrics and lower FPR.
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/Summary.png?raw=true)

Our two-level framework offers flexibility in the way it
is built. Differently from previous works, it goes beyond
applying ensemble learning techniques directly to the
datasets, it uses the predictions obtained to generate four new
meta-datasets based on classes and prediction probabilities
which is applied to all the models used in Level 00
and ensemble learning techniques. This setup provides an
extensive evaluation of the three datasets used, among
them the RoEduNet-SIMARGL2021 which is considered
real-world and not yet evaluated in this context from the
literature gathered to the best of our knowledge. Moreover,
it permits the use of feature selection techniques at both
levels, enriching the pool of results and insights gathered.
Moreover, this work presents an extensive evaluation that is
not present in other works, we analyze 21 models/ensemble
methods in 6 different setups (two Level 00 setups and four
Level 01 setups) in three different datasets, generating results
for Accuracy, Precision, Recall, F1, Runtime, False Positive
Rates, and a Statistical Significance test. Plus, we achieve
perfect and near-perfect results for a few models considering
the FPR and the F1 score, both metrics are particularly crucial
for IDS. This is because security analysts, stakeholders and users need to do everything in their power to identify
a possible threat accurately and as fastest as possible as
undetected attacks can cause significant damage.
We also want to stress that we took the extra step and we
made the codes open source. The way they are built is to
be easily expandable to use with other datasets and further
analysis. As a side note, we want to express that when we say
Framework, we are specifically meaning that our program is
an end-to-end solution delineated by our low-level framework
(Figure 2) that starts at the datasets, then processing all the
inner work and extracting the results. It is not a deployable
solution for production since it is not extensively tested or
validated by an auditory company, but instead, a proof of
concept of the benefits of using our proposed framework and
a crucial stepping-stone in enhancing the field of AI-based
network IDS.

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
