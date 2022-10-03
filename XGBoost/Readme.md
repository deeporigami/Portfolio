# XGBoost Hyperparameter Tuning Projects:

## Introduction

Since Boosting Machine has a tendency of overfitting, XGBoost has an intense focus on addressing "bias-variance trade-off" and facilitates the users to apply a variety of regularization techniques through hyperparameter tuning.

Our objective here is to perform hyperparameter tuning of the native XGBoost API in order to improve its regression performance while addressing "bias-variance trade-off" - especially to alleviate the risk of overfitting. 
In order to conduct hyperparameter tuning, this analysis uses the grid search method. In other words, we select the search grid for hyperparameters and calculate the model performance over all the hyperparameter datapoints on the search-grid. Then, we identify the global local minimum of the performance - or the hyperparameter datapoint which yields the best performance (the minimum value of the Loss Function) - as the best hyperparameter values for the tuned model.

## Computational Constraint

Hyperparameter tuning can be computationally very expensive depending on how you set the search grid. Simply because it needs to iterate performance calculations over all the datapoints determined by the search-grid. The more datapoints you have, the more expensive computationally. Very simple. 

Unfortunately, my notebook has a very limited computational capacity. A good news is that Google Colab provides one GPU per user for a free account. And XGBoost has GPU support feature. Altogether, I can speed up the tuning process using Google Colab's GPU.

All that said, one GPU is still not sufficient when the selected search-grid has an enormous amount of datapoints to cover. Therefore, I still need to address the constraints of the computational resource. In this context, I will perform multiple pair-wise hyperparameter tunings, rather than a single joint simultaneous tuning over all the hyperparameters. This will not only reduce the volume of the hyperparameter datapoints for tuning, but also allow us to render the 3D visualization of the performance landscape for each pair-wise tuning. Of course, there is a catch in this approach.
