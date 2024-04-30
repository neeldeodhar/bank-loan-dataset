#!/usr/bin/env python
# coding: utf-8

# In[51]:


# importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder


# In[52]:


# reading the dataset, printing first 5 rows.
df = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx', sheet_name = 
                  'Data')
print(df.head())


# In[53]:


#viewing the columns info
df.info()


# In[54]:


# view quick statistical measures for continuous attributes:
df[['Age', 'Income', 'CCAvg']].describe()


# In[55]:


# quick measures with filters
df[df['Personal Loan'] ==1][['Age', 'Income','CCAvg']].describe()


# In[56]:


# dealing with missing values
df.isna().sum()


# In[57]:


# dealing with missing values in Mortgage and Family
cols = ['Family', 'Mortgage']
for col in cols:
    df[col] = df[col].fillna(0)
    
df[['Family','Mortgage']].isna().sum()


# In[58]:


#dealing with missing values in age and income
cols = ['Age', 'Income']
for col in cols:
    df[col] = df[col].fillna(df[col].mean())
    
df[['Age', 'Income']].isna().sum()


# In[59]:


# finding duplicate values in ID attribute
sum(df.duplicated(subset = ['ID']))


# In[60]:


# removing duplicate values in ID attribute
df.drop_duplicates(subset = ['ID'], inplace = True)
sum(df.duplicated(subset = ['ID']))


# In[61]:


# dropping unneded columns
df = df.drop(['ID'], axis = 1)
df.head()


# In[62]:


#viewing the ZIP code data
df_county = pd.read_csv("ZIP_County.csv")
df_county.head()


# In[63]:


#merging the ZIP code as a key.
df = pd.merge(df, df_county, how = 'inner', on = 'ZIP Code')
df.head()


# In[64]:


# naming columns
df.columns


# In[65]:


# removing negative values in experience column
print ("Shape Before Filtration: ", df.shape)
df = df[df['Experience']> 0]
print ("Shape After Filtration: ", df.shape)


# In[66]:


# histogram to show distribution as per age
fig = plt.figure(figsize = (6,6))
plt.hist(df['Age'], color = 'r')
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Counts');


# In[67]:


fig = plt.figure(figsize = (6,6))
sns.boxplot(x = df['Age'])
plt.title('Age Box Plot')
plt.xlabel('Age')
plt.ylabel('Box Representation');


# In[68]:


fig = plt.figure(figsize = (6,6))
sns.violinplot(x = df['Age'])
plt.title('Age Violin Plot')
plt.xlabel ('Age')
plt.ylabel('Distribution')


# In[69]:


sns.countplot(x = "Education", data = df)


# In[70]:


sns.countplot(x= 'Education', data = df[df['Personal Loan']== 1])


# In[71]:


#crosstab visualization to show relationship between loans and type of education
pd.crosstab(df['Education'], df['Personal Loan']).plot(kind ="bar", figsize = (6,6))
plt.xlabel('Education')
plt.legend(['No loan', 'Loan'])
plt.ylabel('Number of Occurences')
plt.show();


# In[72]:


#encoding county name
enc = OrdinalEncoder()

df[['County Name']] = enc.fit_transform(df[['County Name']])


# In[73]:


#visualization heatmap

fig = plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True);


# In[74]:


#visualization (pairplot) to show relationships between numerical columns
sns.pairplot(df[['Age', 'Income', 'Experience', 'CCAvg']]);


# In[75]:


# pair plot conclusion from above 
print ("There is a direct strong relationship between age and Experience")


# In[76]:


#my feedback
print ("FEEBACK")
print ("overall a good data analysis with effective use of visualizations")
print ("The visualizations show relationships between attributes and target")


# In[77]:


# correction in the wording
print ("The wording needs to be corrected")
print ("Depositors are asset customers and borrowers are liability customers")
print ("The text has it vice versa")


# In[78]:


#observation on education countplot
print ("The above codes for barplots don't specify type of education categories i.e. undergraduate, graduate , professional etc")
print ("The above codes are only able to label 1, 2 and 3 for each of the categories")

