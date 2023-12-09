#!/usr/bin/env python
# coding: utf-8

# #### Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


# #### Loading IMDb dataset

# In[3]:


imdb = pd.read_csv(r'./IMDb Movies India.csv', encoding='latin-1')
imdb


# #### Checking for missing values

# In[4]:


imdb.info()


# In[5]:


imdb.isna().sum()


# ### EDA

# #### Movies released by Year

# In[6]:


imdb_year = imdb.dropna(subset = ['Year'])
imdb_year['Year'] = imdb_year['Year'].str.replace('(', '').str.replace(')', '')

plt.figure(figsize=(25,10))
plt.title('Number of movies per year', fontsize = 24)
plt.ylabel('Count', fontsize = 18)
plt.xlabel('Year', fontsize = 18)
plt.xticks(fontsize = 14, rotation = 75)
plt.yticks(fontsize = 16)
sns.countplot(data = imdb_year, x = 'Year', order=imdb_year['Year'].value_counts().index)
plt.show()


# #### Highest Rated Movie Ever

# In[7]:


imdb['Rating'].sort_values(ascending = False).nlargest(1).index
imdb.iloc[imdb['Rating'].sort_values(ascending = False).nlargest(1).index]


# #### Top 10 Highest Rated Movies

# In[8]:


imdb.iloc[imdb['Rating'].sort_values(ascending = False).nlargest(10).index]


# #### Top 10 Highest Voted Movies

# In[22]:


imdb['Votes'].sort_values(ascending = False)
top_10_voted = imdb.iloc[imdb['Votes'].sort_values(ascending = False).index[:10]]['Name']
voted_10_df = pd.DataFrame(top_10_voted)
voted_10_names = voted_10_df['Name']
voted_10_df


# In[23]:


print(f'\033[1mThe top 10 most voted movies are: \n\033[0m')
imdb.iloc[voted_10_df.index]


# #### Most Voted Movie per Year

# In[25]:


imdb_year['Votes'] = pd.to_numeric(imdb_year['Votes'].str.replace('[^\d.]', '', regex=True), errors='coerce')
imdb_year.groupby('Year')['Votes'].transform(max)
max_votes_indices = imdb_year.groupby('Year')['Votes'].idxmax().values
max_votes_indices_cln = [x for x in max_votes_indices if str(x) != 'nan']
imdb_year.groupby('Year')['Votes'].idxmax()
most_voted_per_year = imdb.iloc[max_votes_indices_cln][['Name','Year','Genre','Rating','Votes']]
most_voted_per_year['Year'] = most_voted_per_year['Year'].str.replace('(', '').str.replace(')', '')
most_voted_per_year


# In[26]:


# Bubble chart representing most voted movie for each year and size of bubble represents number of votes
# y-axis represents movie rating
fig = px.scatter(most_voted_per_year, x='Year', y='Rating', size='Votes',
                 color='Votes', hover_name='Name', 
                 title='Bubble Chart of Most Voted movies per Year',
                 labels={'Rating': 'Rating'})

fig.update_traces(marker=dict(sizemode='diameter'))  

fig.show()


# In[27]:


# SCALED DOWN Bubble chart representing most voted movie for each year and size of bubble represents number of votes
# y-axis represents movie rating
fig = px.scatter(most_voted_per_year, x='Year', y='Rating', size='Votes',
                 color='Votes', hover_name='Name', 
                 title='Scaled Down Bubble Chart of Most Voted Movies per Year',
                 labels={'Rating': 'Rating'})

size_scaling_factor = 0.2 
most_voted_per_year['ScaledVotes'] = most_voted_per_year['Votes'] * size_scaling_factor


fig.show()


# #### Highest Rated Movie per Year

# In[28]:


# imdb_year.groupby('Year')['R'].transform(max)
highest_rating_indices = imdb_year.groupby('Year')['Rating'].idxmax().values
highest_rating_indices_cln = [x for x in highest_rating_indices if str(x) != 'nan']
imdb_year.groupby('Year')['Rating'].idxmax()
highest_rated_per_year = imdb.iloc[highest_rating_indices_cln][['Name','Year','Genre','Rating','Votes','Duration']]
highest_rated_per_year['Year'] = highest_rated_per_year['Year'].str.replace('(', '').str.replace(')', '')
highest_rated_per_year['Duration'] = highest_rated_per_year['Duration'].str.replace(' min', '')
highest_rated_per_year


# In[32]:


fig = px.bar(highest_rated_per_year, x='Year', y='Rating',
             color='Rating', hover_name='Name',
             title='Highest Rated Movies each Year',
             labels={'Rating': 'Rating'})

fig.update_traces(hovertemplate='Movie Name: %{hovertext}<br>Rating: %{y}')
fig.show()


# #### Movie Rating vs Movie Duration for Highest Rated Movies each Year

# In[36]:


# converting duration to numeric values
highest_rated_per_year['Duration'] = pd.to_numeric(highest_rated_per_year['Duration'], errors='coerce')
# filling missing values
highest_rated_per_year['Duration'].fillna(0, inplace=True)
# scaling Duration
size_scaling_factor = 0.05
highest_rated_per_year['ScaledDuration'] = highest_rated_per_year['Duration'] * size_scaling_factor

# Bubble chart for Movie Rating vs Movie Duration for highest rated movies each year
fig = px.scatter(highest_rated_per_year, x='Year', y='Rating', size='ScaledDuration',
                 color='Duration', hover_name='Name', 
                 title='Bubble Chart of Most Highest Rated per Year',
                 labels={'Rating': 'Rating'})

fig.update_traces(hovertemplate='Movie Name: %{hovertext}<br>Rating: %{y}<br>Duration: %{marker.size}')
fig.show()


# #### Directors with highest number of movies directed and number of movies directed by each

# In[41]:


most_directed_directors = imdb['Director'].value_counts()
most_directed_directors_df = pd.DataFrame(most_directed_directors)
most_directed_directors_df = most_directed_directors_df.reset_index()
most_directed_directors_df.columns = ['Director', 'Number of Movies']
bins = [5, 10, 20, 30, 40, 50, 60]
labels = ['1-10', '10-20', '20-30', '30-40', '40-50', '50-60']
most_directed_directors_df['No of movies range'] = pd.cut(most_directed_directors_df['Number of Movies'], bins=bins, labels=labels)
most_directed_directors_df

fig = px.scatter(most_directed_directors_df, x='Director', y='Number of Movies', color='No of movies range', size='Number of Movies', title='Number of Movies Directed by Each Director (considering those Directors who have directed more than five movies)',
                 labels={'Number of Movies': 'Number of Movies', 'No of movies range': 'Movies Range'})

fig.show()


# #### Directors with highest voted movies 

# In[49]:


most_voted_directors = imdb.groupby('Director')['Votes'].mean().sort_values(ascending = False).index
top_20_most_voted_directors = most_voted_directors[:20].values
accomplished_directors = imdb[imdb['Director'].isin(top_20_most_voted_directors)].sort_values('Votes',ascending = False)
best_movie_per_most_voted_directors = accomplished_directors.loc[accomplished_directors.groupby('Director')['Votes'].idxmax()]
best_movie_per_most_voted_directors = best_movie_per_most_voted_directors.sort_values('Rating')[1:]
best_movie_per_most_voted_directors_sorted = best_movie_per_most_voted_directors.sort_values(by='Votes', ascending=False)
best_movie_per_most_voted_directors_sorted['Year'] = best_movie_per_most_voted_directors_sorted['Year'].str.replace('(', '').str.replace(')', '')

# fig = px.scatter(best_movie_per_most_voted_directors_sorted,
#                  x='Director',
#                  y='Rating',
#                  color='Actor 1',
#                  size='Votes',
#                  title='Movies by Director and Movie Name (Size of Marker Represents Votes)',
#                  labels={'Votes': 'Number of Votes', 'Director': 'Director Name'},
#                  width=1200, height=800)


# plt.figure(figsize=(12, 8))
# sns.scatterplot(x='Votes', y='Director', hue='Actor 1', size='Votes', data=best_movie_per_most_voted_directors_sorted)
# plt.title('Movies by Director and Movie Name (Size of Marker Represents Votes)')
# plt.xlabel('Number of Votes')
# plt.ylabel('Director Name')
# plt.show()

fig = px.bar(best_movie_per_most_voted_directors, x='Director', y='Votes', text='Name',
             title='Movies by Most Accomplished Directors and Movie Name (by Votes)',
             labels={'Director': 'Director Name', 'Votes': 'Number of Votes'})


fig.show()


# #### Highest Rated movies each year with the Highest Votes (Most Popular Movies)

# In[51]:


no_popular_per_year = pd.DataFrame(imdb_year.groupby(['Year', 'Rating', 'Votes']).size())
no_popular_per_year = no_popular_per_year.reset_index()
no_popular_per_year_merged = pd.merge(imdb_year, no_popular_per_year, on=['Year', 'Rating', 'Votes'], how='left')
no_popular_per_year_merged = no_popular_per_year_merged.sort_values(by='Year')
no_most_popular_per_year = no_popular_per_year_merged[(no_popular_per_year_merged['Rating']>8) & (no_popular_per_year_merged['Votes']>2000)]
fig = px.scatter(no_most_popular_per_year, 
                 x='Year', 
                 y='Rating', 
                 size='Votes', 
                 color='Duration',
                 title='Visualization of Most Popular Movies per Year',
                 labels={'Rating': 'Rating', 'Votes': 'Number of Votes'},
                 hover_data=['Name', 'Director', 'Duration', 'Actor 1'])

fig.show()


# #### Data Preparation for Training Predictive Models

# ##### Creating dummies for categorical features

# In[53]:


imdb_year_prep = imdb_year.dropna()


# In[54]:


imdb_year_prep.info()


# In[59]:


genres_dummies = imdb_year['Genre'].str.get_dummies(sep=',')
imdb_year_genre = pd.concat([imdb_year, genres_dummies], axis=1)
director_dummies = imdb_year['Director'].str.get_dummies(sep=',')
imdb_year_genre_director = pd.concat([imdb_year_genre, director_dummies], axis=1)
actor1_dummies = imdb_year['Actor 1'].str.get_dummies(sep=',')
imdb_year_genre_director_a1 = pd.concat([imdb_year_genre_director, actor1_dummies], axis=1)
actor2_dummies = imdb_year['Actor 2'].str.get_dummies(sep=',')
imdb_year_genre_director_a1_a2 = pd.concat([imdb_year_genre_director_a1, actor2_dummies], axis=1)
actor3_dummies = imdb_year['Actor 3'].str.get_dummies(sep=',')
imdb_year_genre_director_a1_a2_a3 = pd.concat([imdb_year_genre_director_a1_a2, actor3_dummies], axis=1)
# dropping original categorical feature column as dummies have been created
imdb_year_genre_director_a1_a2_a3.drop(['Name','Genre','Director','Actor 1','Actor 2', 'Actor 3'], axis = 1, inplace = True)
# dropping rows with "Movie Duration" == 0 
duration_0 = imdb_year_genre_director_a1_a2_a3[imdb_year_genre_director_a1_a2_a3['Duration'] == 0].index
imdb_year_genre_director_a1_a2_a3 = imdb_year_genre_director_a1_a2_a3.drop(duration_0, axis = 0)
# dropping rows with missing values
imdb_year_genre_director_a1_a2_a3 = imdb_year_genre_director_a1_a2_a3.dropna()
# imdb_year_genre_director_a1_a2_a3 = imdb_year_genre_director_a1_a2_a3.drop(['Year','ScaledDuration'], axis = 1)


# In[64]:


# creating instance of min max scaler
scaler = MinMaxScaler()
# declaring predictors and target
X = imdb_year_genre_director_a1_a2_a3.drop('Rating', axis = 1)
y = imdb_year_genre_director_a1_a2_a3['Rating']
# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# normalizing data
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# defining function to train andevaluate models
def models(list_m):
    
    for model in list_m:
        model.fit(X_train_normalized,y_train) 
        y_pred = model.predict(X_test_normalized)
        
        print(f"\033[1m{model} Metrics:\n\033[0m")
        
        print('Mean squared error: ',mean_squared_error(y_test, y_pred))
        print('Mean absolute error: ',mean_absolute_error(y_test, y_pred))
        print('R2 score: ',r2_score(y_test, y_pred))
        print('\n', '='*100, '\n')
        
# declaring algorithms to predict movie rating
model_list = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), lgb.LGBMRegressor()]
models(model_list)


# #### Predicting Target Rating

# In[67]:


lgbm_model = lgb.LGBMRegressor()
lgbm_model.fit(X_train_normalized,y_train)
y_pred = lgbm_model.predict(X_test_normalized)


# In[77]:


df_pred = pd.DataFrame(np.round(y_pred,1), y_test)
df_pred = df_pred.reset_index()
df_pred.columns = ['True Rating', 'Predicted Rating']


# In[78]:


df_pred

