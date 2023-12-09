#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, ConfusionMatrixDisplay, f1_score


# #### Loading IRIS dataset

# In[2]:


iris = pd.read_csv(r'./IRIS.csv', encoding='latin-1')
iris


# #### Checking for missing values

# In[3]:


iris.info()


# #### No NULL values

# ### EDA

# In[4]:


plt.figure(figsize = (8,5))

sns.barplot(data = pd.DataFrame(iris['species'].value_counts()).reset_index(), x = 'index', y = 'species')
plt.ylabel('Count')
plt.xlabel('Species')
plt.title("Counts of Each Species")
plt.show()


# #### Plotting each numeric variable for each species

# In[5]:


sns.pairplot(iris,hue='species',height=3);
plt.show()


# #### Boxplots grouped by Species

# In[6]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

melted_df = pd.melt(iris, id_vars='species', var_name='variable', value_name='value')

variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for variable, ax in zip(variables, axes.flatten()):
    sns.boxplot(x='species', y=variable, data=iris, ax=ax, palette='husl')
    ax.set_title(f'Boxplot of {variable} by Species')
    ax.set_xlabel('Species')
    ax.set_ylabel(variable)

plt.tight_layout()
plt.show()


# #### Heatmap

# In[7]:


iris_corr_mat = iris.corr(numeric_only = True)
plt.figure(figsize = (10,5))
sns.heatmap(iris_corr_mat, annot = True)
plt.show()


# ### Predictive Analysis

# #### Data Preparation

# In[8]:


# Declaring predictor data and target data
X = iris.drop('species', axis = 1)
y = iris['species']
# Splitting the data into training and test set
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Initializing Min Max Scaler
scaler = MinMaxScaler()
# Normalizing data
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)


# #### Model Implementation and Evaluation

# In[9]:


# Defining function for model implementation and Evaluation 
def models(list_m):
    
    for model in list_m:
        model.fit(X_train_normalized,y_train) 
        y_pred = model.predict(X_test_normalized)
        cm = confusion_matrix(y_test, y_pred)
        classes = model.classes_
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        

        print(f"\033[1m{model} Metrics:\n\033[0m")
        
        print("Accuracy Score: {:0.2f}".format(accuracy_score(y_test,y_pred)),'\n')
        print("Weighted F1 Score: {:0.2f}".format(f1_score(y_test, y_pred, average='weighted')),'\n')
        
        print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred),'\n')
        cm_display.plot()
        plt.show()
        print("Classification_Report: \n\n\n",classification_report(y_test,y_pred))


# Choosing various algorithms to perform classification task 
model_list = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

# calling our defined function to fit the training set on various models and perform evaluation for each model
models(model_list)       


# #### Decision Tree is the best performing Classifier with an accuracy Score of 100% and weighted F1 Score of 1

# ### Predicting IRIS species

# In[20]:


final_model =  DecisionTreeClassifier()
# no hyperparamter tuning is done
final_model.fit(X_train_normalized, y_train)
final_pred = final_model.predict(X_test_normalized) 
comparison = pd.DataFrame(final_pred, y_test)
comparison = comparison.reset_index()
comparison.columns = ['Predicted Values', 'True Values']
comparison.set_index('Predicted Values')

