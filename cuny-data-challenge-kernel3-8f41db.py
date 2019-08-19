#############################################################################################################################
###########################################   Given Code Begins  ############################################################
#############################################################################################################################


# ## kernel 3: feature engineering and the hazards of overfitting
# #### Is increasing the power of our model always a good thing?
# 
# 
# 
# As a first step, we're going to import some useful tools and load the data. If this step is unfamiliar to you, try going back to [**kernel_0**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel0).

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

inspections = pd.read_csv('../cuny-data-challenge-2019/inspections_train.csv', parse_dates=['inspection_date'])
x_train0, x_test0 = train_test_split(inspections, test_size=0.75)

# In addition to those tools, we're also going to import **the Random Forest machine learing model**, which is very popular in the data science community (view the documentation [**here**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)). 

# In[2]:

from sklearn.ensemble import RandomForestClassifier

# We'll create some additional features and train a more powerful model. First let's merge the inspections with the data about each venue, and also add the violation counts that we previously calculated.

# In[3]:

# import the venue_stats file
venue_stats = pd.read_csv('../cuny-data-challenge-2019/venues.csv').set_index('camis')

# In[4]:

# import the violations file
violations = pd.read_csv('../cuny-data-challenge-2019/violations.csv', parse_dates=['inspection_date'])

x_train3 = x_train0.merge(venue_stats, 'left', left_on='camis', right_index=True)
x_test3 = x_test0.merge(venue_stats, 'left', left_on='camis', right_index=True)

violation_counts = violations.groupby(['camis', 'inspection_date']).size()
violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])
violation_counts.columns = ['n_violations']

x_train3 = x_train3.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_test3 = x_test3.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
violations.head()
pd.set_option('display.max_colwidth', -1)

# Maybe it would be helpful to include a binary variable that tells us whether the inspection we're looking at is an initial or re-inspection. We can add those easily here.

# In[5]:

x_train3['re_inspect'] = x_train3.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_train3['initial_inspect'] = x_train3.inspection_type.str.contains('initial', regex=False, case=False).map(int)

x_test3['re_inspect'] = x_test3.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_test3['initial_inspect'] = x_test3.inspection_type.str.contains('initial', regex=False, case=False).map(int)

# We might also be interested in knowing which borough the inspection takes place in. From a quick look at the table, I've lumped the six possible options into three categories that roughly correspond to the pass frequency that we see in the training data. Note that this is a way to **encode** categorical features- but not necessarily the smartest way.

# In[6]:

boro_dict = {
    'Missing': 0,
    'STATEN ISLAND': 1,
    'BROOKLYN': 2,
    'MANHATTAN': 3,
    'BRONX': 4,
    'QUEENS': 5    
}

x_train3['boro_idx'] = [boro_dict[_] for _ in x_train3.boro]
x_test3['boro_idx'] = [boro_dict[_] for _ in x_test3.boro]

# Maybe the month of the inspection has some impact on pass frequency? Let's add a variable encoding the month of the year (again, this isn't necessarily the optimal way to encode this into the feature set).

# In[7]:

x_train3['inspection_month'] = (x_train3.inspection_date.dt.strftime('%m').map(int) + 6) % 12
x_test3['inspection_month'] = (x_test3.inspection_date.dt.strftime('%m').map(int) + 6) % 12

# Finally, we'll encode the cuisine description as a numeric variable based on the pass frequency we see corresponding to that cuisine in the training data.

# In[8]:

cuisine_hitrates = x_train3.groupby(['cuisine_description']).agg({'passed':'mean', 'id':'count'}).        rename(columns={'id':'ct'}).sort_values('passed')[['passed']]
cuisine_hitrates.columns = ['cuisine_hr']
    
x_train3 = x_train3.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True)
x_test3 = x_test3.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True)

# We'll make a list of all of the features we've created so that we can pass them to the model for training.

# In[9]:

model_features = ['n_violations', 'inspection_month', 'cuisine_hr', 'boro_idx', 're_inspect', 'initial_inspect']

# We'll use the extremely popular and flexible Random Forest model to generate our predictions. Here we use the default settings of the model to train and generate predictions.

# In[10]:

"""
clf0 = RandomForestClassifier(n_estimators=50)
clf0.fit(x_train3[model_features], x_train3.passed)
test_solution3 = clf0.predict_proba(x_test3[model_features])
loss3a = log_loss(x_test3.passed.values, test_solution3)
print(f'log loss: {loss3a:.3f}')
"""

# Surprisingly, our results are terrible. We've actually done worse than just guessing $0.67$ for every answer. This is the result of overfitting, and is an extremely common issue in machine learning and data science tasks. How do we fix it? We **regularize** the model. In this case, I'll set a maximum depth parameter and a minimum samples per leaf parameter, which prevents the model from getting too complicated. Hopefully this means that it will have predictions that **generalize** to the test data better, rather than overfitting our training data.

# In[11]:

"""
clf1 = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10)
clf1.fit(x_train3[model_features], x_train3.passed)
test_solution4 = clf1.predict_proba(x_test3[model_features])
loss3b = log_loss(x_test3.passed.values, test_solution4)
print(f'log loss: {loss3b:.3f}')
"""

# That's a significant improvement! By make our model a bit less powerful, we beat our best score. 

# Want to try your own parameters? Sometimes it's helpful to store them as a dictionary to keep things organized. Try tweaking these values and training the model again!

# ### Submitting our solution
# In this kernel we've developed a new way to generate solutions. Now we need to generate solutions for each row in the test data, which we find in inspections_test.csv. The steps are:


# In[13]:


"""
# change these parameters!
parameters = {
    'n_estimators': 50,
    'max_depth': 10,
    'min_samples_leaf': 10
}

# we use the ** operator to expand the parameters dictionary
clf_custom = RandomForestClassifier(**parameters)
clf_custom.fit(x_train3[model_features], x_train3.passed)
test_solution_cusotm = clf_custom.predict_proba(x_test3[model_features])
loss3_custom = log_loss(x_test3.passed.values, test_solution_cusotm)
print(f'log loss: {loss3_custom:.3f}')
print('\nX_train3: ', x_train3.shape, '\nX_test3: ', x_test3.shape, '\ntest_solution: ', test_solution_cusotm.shape)
"""


# In[14]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

viol_desc = violations.violation_description
viol_str = viol_desc.str.cat(sep = ' ')
stop = set(stopwords.words('english'))
list_of_words = [i.lower() for i in wordpunct_tokenize(viol_str) if i.lower() not in stop and i.isalpha()]
wordfreqdist = nltk.FreqDist(list_of_words)
mostcommon = wordfreqdist.most_common(30)
combinedDF = pd.merge(violations,inspections[['camis','passed']], right_on = 'camis', left_on = 'camis')
passedDF = combinedDF[combinedDF['passed'] == 1]
failedDF = combinedDF[combinedDF['passed'] == 0]
viol_desc_passed = passedDF.violation_description
viol_desc_failed = failedDF.violation_description
viol_str_passed = viol_desc_passed.str.cat(sep = ' ')
viol_str_failed = viol_desc_failed.str.cat(sep = ' ')
list_of_words_passed = [i.lower() for i in wordpunct_tokenize(viol_str_passed) if i.lower() not in stop and i.isalpha()]
list_of_words_failed = [i.lower() for i in wordpunct_tokenize(viol_str_failed) if i.lower() not in stop and i.isalpha()]
wordfreqdistpassed = nltk.FreqDist(list_of_words_passed)
mostcommonpassed = wordfreqdistpassed.most_common(30)
wordfreqdistfailed = nltk.FreqDist(list_of_words_failed)
mostcommonfailed = wordfreqdistfailed.most_common(30)
failedwordlen = len(list_of_words_failed)
worddictfailed = dict(wordfreqdistfailed)
worddictfailednormalized = {k: float(v) / failedwordlen for k, v in worddictfailed.items()}
passedwordlen = len(list_of_words_passed)
worddictpassed = dict(wordfreqdistpassed)
worddictpassednormalized = {k: float(v) / passedwordlen for k, v in worddictpassed.items()}
worddictrelative = {k: worddictfailednormalized[k] - worddictpassednormalized[k] 
                    for k in worddictfailednormalized if k in worddictpassednormalized}


#############################################################################################################################
###########################################   Given Code Ends  ##############################################################
#############################################################################################################################
# In[15]:


viol_desc_list = viol_desc.values.tolist()
viol_desc_list = [str(x).lower() for x in viol_desc_list]

def score_summer(desc, word_dict):
    word_sum = 0
    for word in word_dict:
        if word in desc:
            to_add = word_dict[word]
            word_sum = word_sum + to_add
    return word_sum


def word_parser(desc_list, word_dict):
    sum_list = []
    for sent in desc_list:
        word_sum = score_summer(sent, word_dict)
        sum_list.append(word_sum)
    return sum_list
        
score = word_parser(viol_desc_list, worddictrelative)

violations['violation_score'] = pd.Series(score)


# In[16]:


business_score = violations.groupby(['camis', 'inspection_date'])['violation_score'].sum()
business_score = business_score.to_frame().reset_index()
business_score.index.name = 'camis'

# In[17]:

x_train3 = x_train3.merge(business_score, on=['camis', 'inspection_date'], how='left')
x_test3 = x_test3.merge(business_score, on=['camis', 'inspection_date'], how='left')


# In[18]:


model_features.append('violation_score')
model_features

# In[23]:


ext_words = ['flies', 'vermin', 'harborage','mice', 'live', 'filth', 'refuse', 'sewage', 'rats', 'âºf', 'roaches']


# In[24]:


def word_check(sent, ext_words):
    word_list = []
    for word in ext_words:
        if word in sent:
            word_list.append(word)
    return word_list

def ext_word_parse(desc_list, ext_words):
    word_list = []
    for sent in desc_list:
        sent_list = word_check(sent, ext_words)
        word_list.append(sent_list)
    return word_list

ext_words_list = ext_word_parse(viol_desc_list, ext_words)

violations['ext_words'] = pd.Series(ext_words_list)


# In[25]:

ext_words_df = violations.groupby(['camis', 'inspection_date'])['ext_words'].apply(lambda x: [item for sublist in x for item in sublist])
ext_words_df = ext_words_df.to_frame().reset_index()
ext_words_df.index.name = 'camis'

# In[28]:


x_train3 = x_train3.merge(ext_words_df, on=['camis', 'inspection_date'], how='left')
x_test3 = x_test3.merge(ext_words_df, on=['camis', 'inspection_date'], how='left')


# In[29]:

# In[30]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
x_train3 = x_train3.join(pd.DataFrame(mlb.fit_transform(x_train3.pop('ext_words')),
                          columns=mlb.classes_,
                          index=x_train3.index))
x_test3 = x_test3.join(pd.DataFrame(mlb.fit_transform(x_test3.pop('ext_words')),
                          columns=mlb.classes_,
                          index=x_test3.index))
x_train3


# In[31]:


for word in ext_words:
    model_features.append(word)
model_features


# In[32]:


from xgboost import XGBClassifier
import time
start_time = time.time()

import datetime

currentDT = datetime.datetime.now()
print(currentDT.strftime("%I:%M:%S %p"))

X = x_train3[model_features]
y = x_train3.passed
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)


xgbc_model = XGBClassifier(n_estimators=2500, max_depth=6, learning_rate=.01, n_jobs=-1, cv=10)
xgbc_model.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='logloss',
                    verbose=False)
prediction = xgbc_model.predict_proba(X_valid)
result = xgbc_model.evals_result()
xgbcloss = log_loss(y_valid, prediction)

print((time.time() - start_time)/60,': ', f'log loss: {xgbcloss:.3f}')

    
# Best so far: .334-.337 with n_est: 1450, learn_r: .02
# Best so far (7/14): .237-.241 with updated violations_score
# 35050, 0.18083770657518244
# 35100, 0.18083742930751404
# Best so far (7/30): .146-.148 with added one hot encode on extreme words, 
# .122-.124 with adjusted percentage of split
# Back to .243


# In[33]:

#############################################################################################################################
###########################################   Given Code Begins  ############################################################
#############################################################################################################################

# load the test data
test_data = pd.read_csv('../cuny-data-challenge-2019/inspections_test.csv', parse_dates=['inspection_date'])

# replicate all of our feature engineering for the test data
test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
test_data = test_data.merge(venue_stats, 'left', left_on='camis', right_index=True)
test_data['re_inspect'] = test_data.inspection_type.str.contains('re-', regex=False, case=False).map(int)
test_data['initial_inspect'] = test_data.inspection_type.str.contains('initial', regex=False, case=False).map(int)
test_data['boro_idx'] = [boro_dict[_] for _ in test_data.boro]
test_data['inspection_month'] = (test_data.inspection_date.dt.strftime('%m').map(int) + 6) % 12
test_data = test_data.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True)
test_data = test_data.merge(business_score.to_frame(), on=['camis', 'inspection_date'], how='left')
test_data = test_data.merge(ext_words_df.to_frame(), on=['camis', 'inspection_date'], how='left')
test_data = test_data.join(pd.DataFrame(mlb.fit_transform(test_data.pop('ext_words')),
                          columns=mlb.classes_,
                          index=test_data.index))


# create a `Predicted` column
# for this example, we're using the model we previously trained
test_data['Predicted'] = [_[1] for _ in xgbc_model.predict_proba(test_data[model_features])]

# take just the `id` and `n_violations` columns (since that's all we need)
submission = test_data[['id', 'Predicted']].copy()

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
submission.to_csv('submission_8_4.csv', index=False)

# let's take a look at our submission to make sure it's what we want
submission.head()

#############################################################################################################################
###########################################   Given Code Ends  ##############################################################
#############################################################################################################################