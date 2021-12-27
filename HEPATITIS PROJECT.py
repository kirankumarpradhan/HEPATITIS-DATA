#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


null_values=["?",]
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data', index_col=False, na_values=null_values,
                    names=["class","AGE","SEX","STEROID","ANTIVIRALS","FATIGUE","MALAISE","ANOREXIA","LIVER BIG","SPLEEN PALPABLE","SPIDERS","ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTINE","HISTOLOGY" ])


# In[9]:


data


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


#replacing missing values
continuous_features = ["AGE","ALK PHOSPHATE",	"SGOT",	"ALBUMIN"	,"PROTINE"]
for column in continuous_features:
    data[column]=data[column].fillna(data[column].mean())

for column in data.columns.drop(continuous_features):
    data[column]=data[column].fillna(data[column].mode().sample(1,random_state=1).values[0])


# In[14]:


#describing data
data.describe()


# In[15]:


plt.figure(figsize=(12,6))
sns.countplot(data['class'])
plt.show()


# In[16]:


plt.figure(figsize=(14,5))
sns.pairplot(data)
plt.show()


# In[18]:


plt.figure(figsize=(14,5))
sns.scatterplot(data['ALK PHOSPHATE'],data['AGE'])
plt.show()


# In[19]:


plt.figure(figsize=(12,6))
sns.barplot(data['class'],data['VARICES'])
plt.show()


# In[20]:


plt.figure(figsize=(12,6))
sns.barplot(data['class'],data['ALK PHOSPHATE'])
plt.show()


# In[21]:


#heat map
corr=data.corr()
plt.subplots(figsize =(16,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[22]:


data.corr()['class'].sort_values()


# In[23]:


#correlation
data.corr()


# In[24]:


X=data.iloc[:,1:19]
y=data.iloc[:,0]


# In[25]:


X.head()


# In[26]:


y.head()


# In[27]:


#splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[28]:


#logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 45, max_iter=1000)
classifier.fit(X_train,y_train)


# In[29]:


#test set results
y_pred = classifier.predict(X_test)


# In[30]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[31]:


print(cm)


# In[32]:


#accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




