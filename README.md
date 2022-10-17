# Credit_Risk_Analysis

## Project Overview

This purpose of this project is to use supervised machine learning models for a company trying to better issue credit card loans. The goal was to predict high risk credit with a high accuracy and with a level of sensitivity and precision that would not allow for great creditors to be declined for bias. Oversampling, undersampling, a combination of the two, plus a bias reducing model were the tecniques used to evaluate credit risk. 

## Machine Learning Models

Naive Random Oversampling
SMOTE Oversampling
Cluster Centroid Undersampling
SMOTEENN Sampling
Balanced Random Forest Classifying
Easy Ensemble Classifying

## Results
## Deliverable 1: Use Resampling Models to Predict Credit Risk

### Random Oversampler

<img width="1097" alt="17RandomOversampling" src="https://user-images.githubusercontent.com/103767830/196263589-62360377-6ac9-4bd5-8031-91a65ed5e4e1.png">

* Balanced Accuracy Score: 0.64
* Precision Scores:
  * High Risk: 0.01
  * Low Risk: 1.00
* Recall Scores:
  * High Risk: 0.66
  * Low Risk: 0.61

### SMOTE

<img width="1097" alt="17SMOTE" src="https://user-images.githubusercontent.com/103767830/196263590-d4d9fd26-c465-46b7-9be4-dd17db1c24c0.png">

* Balanced Accuracy Score: 0.66
* Precision Scores:
  * High Risk: 0.01
  * Low Risk: 1.00
* Recall Scores:
  * High Risk: 0.62
  * Low Risk: 0.69

### Cluster Centroids

<img width="1097" alt="17ClusterCentroids" src="https://user-images.githubusercontent.com/103767830/196263593-0474aaea-79cb-4d0c-beba-ac6a73bb315b.png">

* Balanced Accuracy Score: 0.66
* Precision Scores:
  * High Risk: 0.01
  * Low Risk: 1.00
* Recall Scores:
  * High Risk: 0.69
  * Low Risk: 0.40

## Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk

