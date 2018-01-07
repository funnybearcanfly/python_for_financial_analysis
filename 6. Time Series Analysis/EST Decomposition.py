
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


airline = pd.read_csv('airline_passengers.csv',index_col='Month')


# In[3]:


airline.head()


# In[4]:


airline.plot()


# In[5]:


airline.dropna(inplace=True)


# In[7]:


airline.index = pd.to_datetime(airline.index)


# In[9]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[10]:


result = seasonal_decompose(airline['Thousands of Passengers'],model='multiplicative')


# In[12]:


result.seasonal.plot()


# In[13]:


result.trend.plot()


# In[15]:


fig = result.plot()

