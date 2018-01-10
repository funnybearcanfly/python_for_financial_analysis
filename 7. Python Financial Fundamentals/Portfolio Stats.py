
# coding: utf-8

# In[1]:


import pandas as pd
import quandl


# In[2]:


start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')


# In[3]:


aapl = quandl.get('WIKI/AAPL.11',start_date=start,end_date=end)


# In[4]:


cisco = quandl.get('WIKI/CSCO.11',start_date=start,end_date=end)
ibm = quandl.get('WIKI/IBM.11',start_date=start,end_date=end)
amzn = quandl.get('WIKI/AMZN.11',start_date=start,end_date=end)


# In[7]:


for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close'] / stock_df.iloc[0]['Adj. Close']


# In[9]:


# 30% in apple
# 20% in cisco
# 40% in ibm
# 10% in amazon


# In[11]:


for stock_df, allo in zip((aapl,cisco,ibm,amzn), [.3,.2,.4,.1]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo


# In[13]:


for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Position Values'] = stock_df['Allocation']*1000000


# In[15]:


all_pos_vals = [aapl['Position Values'],cisco['Position Values'],ibm['Position Values'],amzn['Position Values']]
portfolio_val = pd.concat(all_pos_vals,axis=1)


# In[17]:


portfolio_val.columns = ['AAPL Pos', 'CISCO Pos', 'IBM Pos', 'AMZN Pos']


# In[18]:


portfolio_val['Total Pos'] = portfolio_val.sum(axis=1)


# In[20]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[21]:


portfolio_val['Total Pos'].plot()


# In[23]:


portfolio_val.drop('Total Pos',axis=1).plot()


# In[24]:


# Portfolio Stats


# In[25]:


portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)


# In[27]:


daily_return_mean = portfolio_val['Daily Return'].mean()


# In[28]:


daily_return_std = portfolio_val['Daily Return'].std()


# In[29]:


cumulative_return = 100 * (portfolio_val['Total Pos'][-1]/portfolio_val['Total Pos'][0] - 1)


# In[30]:





# In[31]:


SR = daily_return_mean / daily_return_std


# In[34]:


ASR = (252**0.5) * SR


# In[33]:




