
# coding: utf-8

# # Pandas Exercises

# Time to test your new pandas skills! Use the two csv files in this folder to complete the tasks in bold below!
# 
# ** NOTE: ALL TASKS MUST BE DONE IN ONE LINE OF PANDAS CODE. GOT STUCK? NO PROBLEM! CHECK OUT THE SOLUTIONS LECTURE! **

# ** Import pandas and read in the banklist.csv file into a dataframe called banks. **

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('banklist.csv')


# ** Show the head of the dataframe **

# In[3]:





# In[15]:


df.head


# ** What are the column names? **

# In[6]:





# In[7]:


df.columns


# ** How many States (ST) are represented in this data set? **

# In[ ]:


# CODE HERE


# In[9]:


df['ST'].nunique()


# ** Get a list or array of all the states in the data set. **

# In[ ]:


# CODE HERE


# In[10]:


df['ST'].unique()


# ** What are the top 5 states with the most failed banks? **

# In[20]:


# CODE HERE


# In[21]:


df.groupby('ST').count().sort_values('Bank Name',ascending=False).iloc[:5]['Bank Name']


# ** What are the top 5 acquiring institutions? **

# In[ ]:


# CODE HERE


# In[25]:


df.groupby('Acquiring Institution').count().sort_values('Bank Name',ascending=False).iloc[:5]['Bank Name']


# ** How many banks has the State Bank of Texas acquired? How many of them were actually in Texas?**

# In[ ]:


# CODE HERE


# In[26]:


df[df['Acquiring Institution']=='State Bank of Texas']


# ** What is the most common city in California for a bank to fail in?**

# In[ ]:


# CODE HERE


# In[45]:


df[df['ST']=='CA'].groupby('City').count().sort_values('Bank Name',ascending=False).head(1)


# ** How many failed banks don't have the word "Bank" in their name? **

# In[ ]:


# CODE HERE


# In[47]:


sum(df['Bank Name'].apply(lambda name:'Bank' not in name))


# ** How many bank names start with the letter 's' ? **

# In[ ]:


# CODE HERE


# In[48]:


sum(df['Bank Name'].apply(lambda name:name[0].lower()=='s'))


# ** How many CERT values are above 20000 ? **

# In[61]:


# CODE HERE


# In[54]:


sum(df['CERT']>20000)


# ** How many bank names consist of just two words? (e.g. "First Bank" , "Bank Georgia" )**

# In[65]:


# CODE HERE


# In[57]:


sum(df['Bank Name'].apply(lambda name:len(name.split())==2))


# **Bonus: How many banks closed in the year 2008? (this is hard because we technically haven't learned about time series with pandas yet! Feel free to skip this one!**

# In[ ]:


# CODE HERE


# In[59]:


sum(df['Closing Date'].apply(lambda date:date[-2:]=='08'))


# # GREAT JOB!
