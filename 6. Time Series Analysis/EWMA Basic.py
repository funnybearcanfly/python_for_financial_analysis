
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


airline = pd.read_csv('airline_passengers.csv',index_col='Month')


# In[5]:


airline.dropna(inplace=True)


# In[6]:


airline.index = pd.to_datetime(airline.index)


# In[9]:


airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()


# In[10]:


airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()


# In[12]:


airline['EWMA-12'] = airline['Thousands of Passengers'].ewm(span=12).mean()


# In[13]:


airline[['Thousands of Passengers','EWMA-12']].plot()

