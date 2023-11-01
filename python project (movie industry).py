#!/usr/bin/env python
# coding: utf-8

# In[36]:


#import libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[37]:


df = pd.read_csv(r'/Users/hifzasaleem/Downloads/movies.csv')


# In[6]:


df.head()


# In[24]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,pct_missing))


# In[12]:


df = df.dropna(subset=['budget','company'])


# In[13]:


df.head()


# In[14]:


df.dtypes


# In[59]:


#changing data types of columns 
df = df.dropna(subset=['budget', 'gross']).astype({'budget': 'int64', 'gross': 'int64'})


# In[16]:


df.head()


# In[58]:


import numpy as np

# Extract the year and replace NaN with a default value (e.g., 0)
df['yearcorrect'] = df['released'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(0).astype(int)



# In[57]:


df.head()


# In[40]:


df = df.sort_values(by=['gross'],inplace=False, ascending=False)


# In[20]:


pd.set_option('display.max_rows', None)


# In[21]:


df.head()


# In[22]:


plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')
plt.xlabel('gross Earning')
plt.ylabel('Budget for films')
plt.show()


# In[23]:


df.head()


# In[24]:


# plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color": "red"},line_kws={"color":"blue"})


# In[61]:


df.corr(numeric_only=True)


# In[62]:


correlation_matrix = df.corr(method ='pearson', numeric_only=True)

sns.heatmap(correlation_matrix, annot= True)

plt.title('Correlation Matric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[27]:


df.head()


# In[31]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] =  df_numerized[col_name].astype('category')
        df_numerized[col_name] =  df_numerized[col_name].cat.codes

df_numerized.head(20)


# In[41]:


df.head(20)


# In[43]:


correlation_matrix = df_numerized.corr(method ='pearson')

sns.heatmap(correlation_matrix, annot= True)

plt.title('Correlation Matric for Numeric Features')

plt.show()


# In[44]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()

corr_pairs


# In[45]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[53]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[54]:


# votes and budget has highest correlation to gross earniongs 

