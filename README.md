# A Two-Level Ensemble Learning Framework for Enhancing Network Intrusion Detection Systems

In construction... :)    
![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/framework.png?raw=true)

![image](https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/blob/main/images/low_level_framework.png?raw=true)

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
 
    
