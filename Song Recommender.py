
# # Load and explore data

# In[1]:


from sklearn import metrics


# In[1]:


import turicreate as tc


# In[2]:


song_data = tc.SFrame('song_data.gl')


# In[3]:


song_data.head()


# In[4]:


song_data['song'].show()


# In[5]:


len(song_data)


# # Select unique users

# In[6]:


users = song_data['user_id'].unique()

type(users)


# In[8]:


len(users)


# # Creating song recommenders

# In[9]:


train_data, test_data = song_data.random_split(fraction=0.8, seed=0)


# ### 1) Based on popularity

# In[10]:


popularity_model = tc.popularity_recommender.create(train_data, item_id='song', user_id='user_id')


# ### Recommend some songs

# In[11]:


popularity_model.recommend(users=[users[0]])

popularity_model.recommend(users=[users[10]])


# ## 2) Personalised model based on songs similar to what the user has listened to

# In[13]:


personalised_model = tc.item_similarity_recommender.create(train_data, item_id='song', user_id='user_id')


# ### Personalised recommendations

# In[14]:


personalised_model.recommend(users=[users[0]])


# In[15]:


personalised_model.recommend(users=[users[10]])


# ### People that like this also like these....

# In[16]:


personalised_model.get_similar_items(items=['Naked - Marques Houston'])


# In[17]:


personalised_model.get_similar_items(items=['Nice & Slow - Usher'])


# # Comparing model performance

# ### We will use AUC (area under curve) of precision vs recall

# In[18]:


model_performance = tc.recommender.util.compare_models(test_data, models=[popularity_model,personalised_model], model_names=['popularity','personalised'], user_sample=0.01)
# We use 0.05% of the sample because it takes very long


type(model_performance)


# In[20]:


len(model_performance)


# In[21]:


model_performance[0]


# In[22]:


model_performance[1]

type(model_performance[0])


# In[24]:


len(model_performance[0])


# In[25]:


model_performance[0].keys()


model_performance[0]['precision_recall_overall']



model_performance[1]['precision_recall_overall']


# ## Plot Precision-Recall Curves and calculate AUCs

# In[31]:


type(model_performance[1]['precision_recall_overall']['precision'])


# In[35]:


from sklearn.metrics import auc


# In[37]:


popularity_auc = auc(model_performance[0]['precision_recall_overall']['recall'], model_performance[0]['precision_recall_overall']['precision'])
personalised_auc = auc(model_performance[1]['precision_recall_overall']['recall'], model_performance[1]['precision_recall_overall']['precision'])


# In[38]:


print(popularity_auc, personalised_auc)


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


plt.plot(model_performance[0]['precision_recall_overall']['recall'], 
         model_performance[0]['precision_recall_overall']['precision'], 
         '.',
         label='popularity (AUC=%0.3f)' % popularity_auc)
plt.plot(model_performance[1]['precision_recall_overall']['recall'], 
         model_performance[1]['precision_recall_overall']['precision'], 
         '-',
         label='personalised (AUC=%0.3f)' % personalised_auc)
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

