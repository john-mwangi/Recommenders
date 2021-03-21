
# Use 500K ratings instead of 20M (pilot)
# Automatically recommend and appropriate model and retrieve its type
# Incorporate product meta-data
# Random split data by user
# Evaluate precision/recall and rmse
# Query the model to return movie names & similar items
# Save variables using Dill


import turicreate as tc


# In[3]:


ratings = tc.SFrame.read_csv('ratings.csv')
movies = tc.SFrame.read_csv('movies.csv')


# In[4]:


ratings.head()


# In[5]:


len(ratings)


# # Randomly select 500K ratings

# In[6]:


500000/27753444


# In[4]:


ratings_2, ratings_3 = ratings.random_split(fraction=0.018015782113383838, seed=0)


# In[5]:


len(ratings_2)


# # Select matching movie ratings

# In[6]:


movies_2 = ratings_2['movieId'].unique()


# In[7]:


len(movies_2), len(movies)


# In[15]:


movies.head()


# In[8]:


movies_2 = movies.filter_by(column_name='movieId', values=movies_2)


# In[9]:


print(type(movies_2), movies_2.shape)


# # Training & testing datasets splitting

# In[10]:


train_data, test_data = tc.recommender.util.random_split_by_user(dataset=ratings_2, item_id='movieId', item_test_proportion=0.2, random_seed=0, user_id='userId')
# This is a random split that is specially designed for recommenders


# In[11]:


print(ratings_2.shape, train_data.shape, test_data.shape)


# In[26]:


723/499353

# # Train the model while incorporating meta-data

# In[12]:


model_1 = tc.recommender.create(observation_data=train_data,
                                user_id='userId',
                                item_id='movieId',
                                item_data=movies_2,
                                target='rating')
# This automatically selects the best recommender based on the data fed
# target is optional to take care of situations where customers rate products


# # Get type and description of model selected

# In[13]:


model_1.__str__()
# For song recommender we used item_similarity_recommender (with no targets)


# In[30]:


print(model_1.__repr__())


# # Evaluate PR & RMSE

# ### 1) RMSE

from sklearn.metrics import mean_squared_error
from math import sqrt


# In[15]:


preds = model_1.predict(dataset=test_data)
rmse = sqrt(mean_squared_error(y_pred=preds, y_true=test_data['rating']))


# In[16]:


rmse


# ### 2) Precision/Recall

# In[17]:


precision_recall = model_1.evaluate_precision_recall(dataset=test_data)


# In[33]:


precision_recall


# In[43]:


type(precision_recall)


# In[44]:


len(precision_recall)


# In[45]:


precision_recall.keys()


# In[46]:


precision_recall['precision_recall_overall']


# In[18]:


from sklearn.metrics import auc


# In[19]:


auc(x=precision_recall['precision_recall_overall']['recall'],
    y=precision_recall['precision_recall_overall']['precision'])


# # Use model to recommend products

# In[20]:


users = ratings_2['userId'].unique()


# In[22]:


len(users), type(users)


# In[23]:


model_1.recommend(users=[users[0]])


# In[39]:


model_1.get_similar_items(items=[318])


# In[30]:


similar = model_1.get_similar_items()


# In[37]:


similar[similar['movieId'] == 318]


# # Save variables

# In[40]:


import dill
import pickle


# In[41]:


dill.dump_session('movie_recommender_08Feb')


# In[9]:


get_ipython().system('jupyter nbconvert --to python "Movie Recommender.ipynb"')


# In[ ]:




