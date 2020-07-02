#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[12]:


# closed form solution
def simple_linear_regression_closed_form(feature, target):
    X = feature.to_numpy()
    y = target.to_numpy()
    
    N = X.shape[0]
    
    slope_num = np.dot(X, y) + (np.sum(X) * np.sum(y)) / N
    slope_den = np.dot(X, X) + (np.sum(X) ** 2) / N
    
    slope = slope_num / slope_den
    intercept = np.sum(y) / N - slope * np.sum(X) / N
    
    return (intercept, slope)


# In[15]:


def get_prediction(feature, coeff):
    y_pred = coeff[1] * feature + coeff[0]
    return y_pred


# In[6]:


def get_inverse_prediction(target, coeff): 
    X_pred = (target - coeff[0]) / coeff[1]
    return X_pred


# In[7]:


def get_rss(feature, target, coeff):
    rss = np.sum((coeff[0] + coeff[1] * feature - target) ** 2)
    return rss


# In[ ]:




