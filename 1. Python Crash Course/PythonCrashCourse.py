
# coding: utf-8

# # Python Crash Course Exercises - Solutions
# 
# This is an optional exercise to test your understanding of Python Basics. The questions tend to have a financial theme to them, but don't look to deeply into these tasks themselves, many of them don't hold any significance and are meaningless. If you find this extremely challenging, then you probably are not ready for the rest of this course yet and don't have enough programming experience to continue. I would suggest you take another course more geared towards complete beginners, such as [Complete Python Bootcamp]()

# ## Exercises
# 
# Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

# ### Task #1
# 
# Given price = 300 , use python to figure out the square root of the price.

# In[3]:


price = 300


# In[4]:


price**0.5


# In[6]:


import math
math.sqrt(price)


# ### Task #2
# 
# Given the string:
# 
#     stock_index = "SP500"
#    
# Grab '500' from the string using indexing.

# In[1]:


stock_index = "SP500"


# In[2]:


stock_index[2:]


# ### Task #3
# 
# ** Given the variables:**
# 
#     stock_index = "SP500"
#     price = 300
# 
# ** Use .format() to print the following string: **
# 
#     The SP500 is at 300 today.

# In[7]:


stock_index = "SP500"
price = 300


# In[9]:


print("The {} is at {} today.".format(stock_index,price))


# ### Task #4
# 
# ** Given the variable of a nested dictionary with nested lists: **
# 
#     stock_info = {'sp500':{'today':300,'yesterday': 250}, 'info':['Time',[24,7,365]]}
#     
# ** Use indexing and key calls to grab the following items:**
# 
# * Yesterday's SP500 price (250)
# * The number 365 nested inside a list nested inside the 'info' key.

# In[10]:


stock_info = {'sp500':{'today':300,'yesterday': 250}, 'info':['Time',[24,7,365]]}


# In[12]:


stock_info['sp500']['yesterday']


# In[15]:


stock_info['info'][1][2]


# ### Task #5
# 
# ** Given strings with this form where the last source value is always separated by two dashes -- **
# 
#     "PRICE:345.324:SOURCE--QUANDL"
#     
# **Create a function called source_finder() that returns the source. For example, the above string passed into the function would return "QUANDL"**

# In[16]:


def source_finder(s):
    return s.split('--')[-1]


# In[18]:


source_finder("PRICE:345.324:SOURCE--QUANDL")


# ### Task #5
# 
# ** Create a function called price_finder that returns True if the word 'price' is in a string. Your function should work even if 'Price' is capitalized or next to punctuation ('price!')  **

# In[19]:


def price_finder(s):
    return 'price' in s.lower()


# In[20]:


price_finder("What is the price?")


# In[22]:


price_finder("DUDE, WHAT IS PRICE!!!")


# In[23]:


price_finder("The price is 300")


# ### Task #6
# 
# ** Create a function called count_price() that counts the number of times the word "price" occurs in a string. Account for capitalization and if the word price is next to punctuation. **

# In[46]:


def count_price(s):
    count = 0
    for word in s.lower().split():
        # Need to use in, can't use == or will get error with punctuation
        if 'price' in word:
            count += 1
            
    # Note the indentation!
    return count


# In[43]:


# Simpler Alternative
def count_price(s):
    return s.lower().count('price')


# In[44]:


s = 'Wow that is a nice price, very nice Price! I said price 3 times.'


# In[47]:


count_price(s)


# ### Task #7
# 
# **Create a function called avg_price that takes in a list of stock price numbers and calculates the average (Sum of the numbers divided by the number of elements in the list). It should return a float. **

# In[27]:


def avg_price(stocks):
    return sum(stocks)/len(stocks) # Python 2 users should multiply numerator by 1.0


# In[ ]:


avg_price([3,4,5])


# # Great job!
