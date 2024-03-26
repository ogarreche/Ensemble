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

  - The accuracy for the AI model.
  - The values for y_axis for the sparsity (that will need to be copied and pasted into the general_sparsity_graph.py).
  - Top features in importance order (that will be needed to rerun these same programs to obtain new Accuracy values for Descriptive Accuracy. Take note of the values as you use less features and input these values in  general_desc_acc_graph.py ).
  - For Stability, run the programs 3x or more and input the obtained top k features in general_stability_comparison.py).
    
