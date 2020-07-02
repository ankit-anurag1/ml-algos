#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


def get_feature_matrix(dataset, features):
    
    # add a constant column to the front of the dataset, with value 1
    # return a numpy.array
    
    return feature_matrix


# In[4]:


def gradient_descent(feature_matrix, output_array, init_weights, eta, tolerance):
    weights = np.array(init_weights)
    converged = False
    
    while not converged:
        # predicted value using current weights
        y_pred = np.dot(feature_matrix, weights)
        
        # gradient of loss
        dJ = 2 * np.dot(feature_matrix.T, output_array - y_pred)
        
        # modifying weights
        weights = weights + (eta * dJ)
        
        if np.sqrt(np.sum(dJ ** 2)) <= tolerance:
            converged = True   
    
    return weights


# In[5]:


def get_predictions(dataset, features, weights):
    
    features = features + ['constant']

    # add a constant column to the front of the dataset, with value 1
    feature_matrix = np.array(dataset[features])
    
    y_pred = np.dot(feature_matrix, weights)
    
    return y_pred


# In[6]:


def get_rss(y_pred, y_actual):
    return np.sum((y_pred - y_actual) ** 2)


# In[ ]:




