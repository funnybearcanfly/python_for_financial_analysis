
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# <center>*Copyright Pierian Data 2017*</center>
# <center>*For more information, visit us at www.pieriandata.com*</center>

# # NumPy Exercises - Solutions
# 
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks and then you'll be asked some more complicated questions.
# 
# ** IMPORTANT NOTE! Make sure you don't run the cells directly above the example output shown, otherwise you will end up writing over the example output! **

# #### Import NumPy as np

# In[1]:


import numpy as np


# #### Create an array of 10 zeros 

# In[ ]:


# CODE HERE


# In[2]:


np.zeros(10)


# #### Create an array of 10 ones

# In[ ]:


# CODE HERE


# In[3]:


np.ones(10)


# #### Create an array of 10 fives

# In[ ]:


# CODE HERE


# In[4]:


np.ones(10) * 5


# #### Create an array of the integers from 10 to 50

# In[ ]:


# CODE HERE


# In[5]:


np.arange(10,51)


# #### Create an array of all the even integers from 10 to 50

# In[ ]:


# CODE HERE


# In[6]:


np.arange(10,51,2)


# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[ ]:


# CODE HERE


# In[7]:


np.arange(9).reshape(3,3)


# #### Create a 3x3 identity matrix

# In[ ]:


# CODE HERE


# In[8]:


np.eye(3)


# #### Use NumPy to generate a random number between 0 and 1

# In[ ]:


# CODE HERE


# In[15]:


np.random.rand(1)


# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[ ]:


# CODE HERE


# In[33]:


np.random.randn(25)


# #### Create the following matrix:

# In[35]:


np.arange(1,101).reshape(10,10) / 100


# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[36]:


np.linspace(0,1,20)


# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[ ]:


# CODE HERE


# In[38]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[39]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[40]:


mat[2:,1:]


# In[29]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[41]:


mat[3,4]


# In[30]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[42]:


mat[:3,1:2]


# In[31]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[46]:


mat[4,:]


# In[32]:


# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE


# In[49]:


mat[3:5,:]


# ### Now do the following

# #### Get the sum of all the values in mat

# In[ ]:


# CODE HERE


# In[50]:


mat.sum()


# #### Get the standard deviation of the values in mat

# In[ ]:


# CODE HERE


# In[51]:


mat.std()


# #### Get the sum of all the columns in mat

# In[ ]:


# CODE HERE


# In[53]:


mat.sum(axis=0)


# ## Bonus Question
# 
# We worked a lot with random data with numpy, but is there a way we can insure that we always get the same random numbers? [Click Here for a Hint](https://www.google.com/search?q=numpy+random+seed&rlz=1C1CHBF_enUS747US747&oq=numpy+random+seed&aqs=chrome..69i57j69i60j0l4.2087j0j7&sourceid=chrome&ie=UTF-8)

# In[1]:


np.random.seed(101)


# # Great Job!
