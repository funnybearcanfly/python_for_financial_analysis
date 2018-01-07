
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


import statsmodels.api as sm


# In[4]:


df = sm.datasets.macrodata.load_pandas().data


# In[5]:


df.head()


# In[6]:


index = sm.tsa.datetools.dates_from_range('1959Q1','2009Q3')


# In[9]:


df.index = index


# In[10]:


df.head()


# In[11]:


df['realgdp'].plot()


# In[15]:


gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp'])


# In[17]:


df['trend'] = gdp_trend
df[['realgdp','trend']]["2000-03-31":].plot()

