
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[5]:


aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
cisco = pd.read_csv('CISCO_CLOSE',index_col='Date',parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE',index_col='Date',parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE',index_col='Date',parse_dates=True)


# In[6]:


stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)


# In[7]:


stocks.columns = ['aapl','cisco','ibm','amzn']


# In[8]:


log_ret = np.log(stocks/stocks.shift(1))


# In[9]:


# Random Optimization
num_ports = 5000

all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    # Weights
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)

    all_weights[ind,:] = weights
    
    # Expected Return
    ret_arr[ind] = np.sum(log_ret.mean()*weights*252)

    # Expected Volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]


# In[10]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilitiy')
plt.ylabel('Return')

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolor='black')


# In[11]:


# Mathmatically Optimization
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])


# In[12]:


from scipy.optimize import minimize


# In[15]:


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1


# In[16]:


def check_sum(weights):
    # return 0 if the sum of the weights is 1
    return np.sum(weights) - 1


# In[17]:


cons = ({'type':'eq','fun':check_sum})


# In[18]:


bounds = ((0,1),(0,1),(0,1),(0,1))


# In[19]:


init_guess = [0.25,0.25,0.25,0.25]


# In[20]:


opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)


# In[21]:





# In[23]:


get_ret_vol_sr(opt_results.x)


# In[24]:


frontier_y = np.linspace(0,0.3,100)


# In[25]:


def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


# In[27]:


frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type':'eq','fun':check_sum},
           {'type':'eq','fun':lambda w: get_ret_vol_sr(w)[0]-possible_return})

    result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    
    frontier_volatility.append(result['fun'])


# In[28]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilitiy')
plt.ylabel('Return')

plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)

