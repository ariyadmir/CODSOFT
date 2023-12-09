#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, ConfusionMatrixDisplay
import imblearn
from imblearn.over_sampling import SMOTE


# ### Loading the dataset

# In[3]:


titanic = pd.read_csv('./tested.csv')


# ### Exploratory Data Analysis

# #### Missing values

# In[5]:


titanic.info()


# In[6]:


titanic.describe()


# #### Dropping feature Cabin as most values are Null
# 

# In[7]:


titanic = titanic.drop('Cabin', axis = 1)


# #### Imputing values of 'Age'
# Filling missing values with most frequently occuring age by each category of sex and class

# In[8]:


most_frequent_age_by_group = titanic.groupby(['Pclass', 'Sex'])['Age'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
most_frequent_age_by_group


# In[9]:


# creating new column for imputed Age
titanic = pd.merge(titanic, most_frequent_age_by_group, on=['Pclass', 'Sex'], how='left', suffixes=('', '_imputed'))
# adding imputed ages back into original dataframe
titanic['Age'].fillna(titanic['Age_imputed'], inplace=True)
# droping the imputed_Age column used for imputation
titanic.drop(['Age_imputed'], axis=1, inplace=True)


# #### Filling missing 'Fare' value 

# In[10]:


mean_fare = titanic['Fare'].mean()
titanic['Fare'].fillna(mean_fare, inplace = True)


# ### EDA

# #### Distribution of Titanic Passengers by 'Sex'

# In[11]:


titanic['Sex'].value_counts()


# In[89]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Titanic Passengers by 'Sex'", fontsize = 16)
plt.ylabel('Count', fontsize = 14)
plt.xlabel('Sex', fontsize = 14)
plt.xticks(fontsize = 14)
ax = sns.countplot(data = titanic, x = 'Sex')
total = len(titanic['Sex'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3,
            '{:.2f}%'.format((height / total) * 100),
            ha="center", fontsize=12)
plt.show()


# #### Distribution of Survival Count of Titanic Passengers by 'Sex'

# In[90]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Survival Count of Titanic Passengers by 'Sex'", fontsize = 16)
plt.ylabel('Count', fontsize = 14)
plt.xlabel('Sex', fontsize = 14)
plt.xticks(fontsize = 14)
ax = sns.countplot(data = titanic, x = 'Survived', hue = 'Sex')
total = len(titanic['Sex'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3,
            '{:.2f}%'.format((height / total) * 100),
            ha="center", fontsize=12)
plt.show()


# ##### The EDA suggests that all FEMALE passengers survived, whereas all MALE passengers did not survive

# #### Distribution of Titanic Passengers by 'Age' and 'Sex'

# In[15]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Titanic Passengers by 'Age'", fontsize = 16)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Age', fontsize = 14)
plt.xticks(fontsize = 14)
sns.histplot(data = titanic, x = 'Age', hue = 'Sex', kde = True)
plt.show()


# In[16]:


age_df = pd.DataFrame(titanic.groupby('Sex')['Age'].mean())
print(f'\033[1mThe mean age of Male passengers is: {age_df.iloc[0][0]}\033[0m')
print(f'\033[1mThe mean age of Female passengers is: {age_df.iloc[1][0]}\033[0m')


# #### Distribution of Survival Count of Titanic Passengers by 'Age'

# In[18]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Survival Count of Titanic Passengers by 'Age'", fontsize = 16)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Age', fontsize = 14)
plt.xticks(fontsize = 14)
sns.histplot(data = titanic, x = 'Age', hue = 'Survived', kde = True)
plt.show()


# #### Correlation between numeric features

# In[19]:


corr = titanic.drop('PassengerId', axis = 1).corr(numeric_only = True)
plt.figure(figsize = (10,5))
sns.heatmap(data = corr, annot = True)
plt.show()


# In[20]:


print(f'\033[1mLower values of Passenger Class (1) is correlated with higher fares; \nHigher values of Passenger Class (3) with lower fares\033[0m')
print(f'\n\n\033[1mLower values of Passenger Class (1) is correlated with higher values of Age.\nUpper Class passengers belong to an older Age range and lower class passengers are younger in Age.\033[0m')


# #### Distribution of Titanic Passengers by Passenger Class and Age

# In[21]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Titanic Passengers by Passenger Class and Age", fontsize = 16)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Age', fontsize = 14)
plt.xticks(fontsize = 14)
sns.histplot(data = titanic, x = 'Age', hue = 'Pclass', kde = True, palette = 'Set1')
plt.show()


# #### Distribution of Passengers by Ticket 'Fare' for each 'Passenger Class'
# 

# In[22]:


plt.figure(figsize=(10,5))
ax = sns.stripplot(data = titanic, x = 'Fare', y = 'Pclass', hue = 'Pclass', size = 6)
ax.set_title("Distribution of Passengers by Ticket 'Fare' for each 'Passenger Class'", fontsize = 16)
ax.set_ylabel('Passenger Class Density', fontsize = 14)
ax.set_xlabel('Fare', fontsize = 14)
xtick_loc = [25,50,75,100,125,150,175,200]
ytick_loc = [1, 2, 3]
ax.set_xticks(xtick_loc)
ax.set_yticks(ytick_loc)
plt.xticks(fontsize = 14)

plt.show()


# Upper Class ticket fare is higher

# ### The Privileged Class!

# In[23]:


print(f"\033[1mInterestingly, {len(titanic[titanic['Fare']==0])} First Class passengers travelled for free.\n\nMr. Roderick Robert Crispin Chisholm (the co-designer of RMS Titanic) and Mr. Joseph Bruce Ismay (the chariman and director of the White Star Line - the company that owned the Titanic)\033[0m")


# In[ ]:





# #### Survival of passengers according to Sex and Class

# In[25]:


combinations = (
    titanic.groupby(['Pclass', 'Sex','Survived'])
    .size()
    .reset_index()
    .rename(columns={0: 'count'})
)

fig = px.sunburst(
    titanic,
    path=['Pclass', 'Sex', 'Survived'],
    title='Survival of passengers according to Sex and Class',
    color='Survived',
    height=500,
)

fig.show()


# #### Distribution of Passengers by number of Siblings/Spouses

# In[26]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Passengers by number of Siblings/Spouses", fontsize = 16)

sns.countplot(data = titanic, x = 'Survived', hue = 'SibSp')
plt.xticks([0,1],['Did not Survive', 'Survived'], fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Passenger Survival', fontsize = 14)
plt.legend(title = 'No. of Siblings/Spouses')
plt.show()


# #### Distribution of Passengers by number of Parents/Children

# In[27]:


plt.figure(figsize=(10,5))
plt.title("Distribution of Passengers by number of Parents/Children", fontsize = 16)

sns.countplot(data = titanic, x = 'Survived', hue = 'Parch')
plt.xticks([0,1],['Did not Survive', 'Survived'], fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Passenger Survival', fontsize = 14)
plt.legend(title = 'No. of Parents/Children')
plt.show()


# #### Creating another feature 'Large Family'
# 1 for passengers that are accompanied by more than 1 Sibling/Spouse/Parent/Child
# 
# 0 for passengers accompanied by 1 or 0 Sibling/Spouse/Parent/Child

# In[28]:


idx = list(titanic[(titanic['SibSp']>1)|(titanic['Parch']>1)].index)
titanic['LargeFamily'] = 0
titanic.loc[idx, 'LargeFamily'] = 1
titanic['LargeFamily'].value_counts()


# #### Passenger survival number according to Family Size

# In[29]:


plt.figure(figsize=(10,5))
plt.title("Passenger survival number according to Family Size", fontsize = 16)

sns.countplot(data = titanic, x = 'Survived', hue = 'LargeFamily')
plt.xticks([0,1],['Did not Survive', 'Survived'], fontsize = 14)
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Passenger Survival', fontsize = 14)
plt.legend(title = 'Large Family:\nNo. of parents/chidlren/siblings/spouse\n', labels = ['<=1', ' >1'], fontsize = 10)
plt.show()


# ##### Passengers travelling alone or with a single companion are more likely to NOT SURVIVE than those travelling with larger families
# 

# #### Passenger Class by Embarking Station

# In[31]:


plt.figure(figsize=(10,5))
plt.title("Passenger class by Embarking Station", fontsize = 16)

sns.countplot(data = titanic,x= 'Embarked',hue='Pclass')
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Embarking Station', fontsize = 14)
plt.show()


# #### Passenger survival by Embarking Station

# In[32]:


plt.figure(figsize=(10,5))
plt.title("Passenger survival by Embarking Station", fontsize = 16)

sns.countplot(data = titanic,x= 'Embarked',hue='Survived')
plt.ylabel('Number of Passengers', fontsize = 14)
plt.xlabel('Embarking Station', fontsize = 14)
plt.show()


# ### Data Preparation for Model Training

# ##### Dropping features that are not relevant predictors

# In[33]:


titanic = titanic.drop('PassengerId', axis = 1)
titanic = titanic.drop('Name', axis = 1)
titanic = titanic.drop('Ticket', axis = 1)


# ##### Defining target and predictors

# In[34]:


X = titanic.drop('Survived', axis = 1)
y = titanic['Survived']


# ##### Splitting data into Training and Testing datasets

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.head()


# #### Encoding categorical features
# 
# 'Pclass', 'Sex', and 'Embarked'

# In[37]:


# Initializing encoder 
encoder = OneHotEncoder(handle_unknown='ignore')


# ##### Encoding 'Sex'

# In[38]:


X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True )
encoded_sex_train = pd.DataFrame(encoder.fit_transform(X_train[['Sex']]).toarray())
encoded_sex_test = pd.DataFrame(encoder.fit_transform(X_test[['Sex']]).toarray())
encoded_sex_train.columns = encoder.get_feature_names_out(['Sex'])
encoded_sex_test.columns = encoder.get_feature_names_out(['Sex'])
X_train_sex_encoded = pd.concat([X_train, encoded_sex_train], axis =1)
X_test_sex_encoded = pd.concat([X_test, encoded_sex_test], axis =1)


# ##### Encoding 'Embarked'

# In[40]:


encoded_embarked_train = pd.DataFrame(encoder.fit_transform(X_train[['Embarked']]).toarray())
encoded_embarked_test = pd.DataFrame(encoder.fit_transform(X_test[['Embarked']]).toarray())
encoded_embarked_train.columns = encoder.get_feature_names_out(['Embarked'])
encoded_embarked_test.columns = encoder.get_feature_names_out(['Embarked'])
X_train_encoded = pd.concat([X_train_sex_encoded, encoded_embarked_train], axis =1)
X_test_encoded = pd.concat([X_test_sex_encoded, encoded_embarked_test], axis =1)


# #### Encoding 'Passenger Class'

# In[447]:


encoded_pclass_train = pd.DataFrame(encoder.fit_transform(X_train[['Pclass']]).toarray())
encoded_pclass_test = pd.DataFrame(encoder.fit_transform(X_test[['Pclass']]).toarray())
encoded_pclass_train.columns = encoder.get_feature_names_out(['Pclass'])
encoded_pclass_test.columns = encoder.get_feature_names_out(['Pclass'])
X_train_encoded = pd.concat([X_train_encoded, encoded_pclass_train], axis =1)
X_test_encoded = pd.concat([X_test_encoded, encoded_pclass_test], axis =1)


# Dropping original categorical columns (after encoding)

# In[41]:


X_train_encoded = X_train_encoded.drop(['Pclass', 'Sex','Embarked'], axis = 1)
X_test_encoded = X_test_encoded.drop(['Pclass', 'Sex','Embarked'], axis = 1)
print('\033[1mX_train_encoded:\033[0m')
display(X_train_encoded.head())
print('\033[1mX_train_encoded:\033[0m')
display(X_test_encoded.head())


# #### Data Normalization (scaling)

# In[43]:


# initializing Min Max Scaler
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_encoded)
X_test_normalized = scaler.transform(X_test_encoded)


# ### Model Implementation and Evaluation
# Choosing algorithms most effective for binary classification: Logistic Regression, Decision Tree Classification, Random Forest Classification, and KNN Classification

# In[85]:


def models(list_m):
    
    for model in list_m:
        model.fit(X_train_normalized,y_train) 
        y_pred = model.predict(X_test_normalized)
        cm = confusion_matrix(y_test, y_pred)
        classes = model.classes_
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        print(f"\033[1m{model} Metrics:\n\033[0m")
        
        print("Accuracy Score: {:0.2f}".format(accuracy_score(y_test,y_pred)),'\n')
        print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred),'\n')
        cm_display.plot()
        plt.show()
        print("Classification_Report: \n\n\n",classification_report(y_test,y_pred))
        


# In[86]:


model_list = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

# calling our defined function to fit the training set on various models and perform evaluation for each model
models(model_list)


# #### Visualizing Confusion Matrix

# Each algorithm predicts the survival of the passengers in the Test dataset with an accuracy of 100%
# 
# It might be that the high accuracy is due to overfitting attributed to class imbalance in the training dataset
# 
# #### Addressing Class Imbalance of Target 'Survived'

# In[62]:


y_train.value_counts()


# In[63]:


sns.countplot(data = titanic, x = 'Survived')


# #### Oversampling with SMOTE

# In[68]:


oversample = SMOTE()
X_train_os, y_train_os = oversample.fit_resample(X_train_normalized, y_train)


# ###### Now the training target variable is balanced

# In[70]:


y_train_os.value_counts()


# ###### Model implementation and evaluation for each of our chosen algorithms using oversampled training data

# In[87]:


X_train_normalized = X_train_os
y_train = y_train_os

models(model_list)


# #### Oversampling using SMOTE has no significant effect on model accuracy. 
# 
# Balanced accuracy for all chosen algorithms = 100%
