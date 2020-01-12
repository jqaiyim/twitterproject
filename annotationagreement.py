
# coding: utf-8

# 

# In[13]:
import statsmodels.api as sm

import os
import pickle
# from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import gzip as gz
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import seaborn as sns

from scipy import stats


# In[2]:

def get_noun_count(text):
    tokens = nltk.pos_tag(text.split())
    return len([token for token in tokens if 
               (token[1] == 'NN' or token[1] == 'NNP' or token[1] == 'NNS' or 
             token[1] == 'NNPS') ])
def get_adjective_count(text):
    tokens = nltk.pos_tag(text.split())
    return len([token for token in tokens if 
               (token[1] == 'JJ' or token[1] == 'JJR' or token[1] == 'JJS')])
def get_adverb_count(text):
    tokens = nltk.pos_tag(text.split())
    return len([token for token in tokens if 
               (token[1] == 'RB' or token[1] == 'RBR' or token[1] == 'RBS')])


# In[3]:

def load_data(file, coder):
    df = pd.read_excel(file, encoding='utf8')
    print('Initial size:', len(df))
    df['tweet-id']=df['tweet-id'].astype(str)    
    df=df[~df['professional'].isnull()]
    print('Labeled:',len(df))
    df['professional']= df.professional.astype(int)
    df.rename(columns={'tweet-id':'tweet_id','professional':coder}, inplace=True)    
    return df[['tweet_id','text',coder,'url']].set_index('tweet_id', drop=True)

def load_data1(file, coder):
    df = pd.read_csv(file)
    print('Initial size:', len(df))
    df['tweet-id']=df['tweet-id'].astype(str)
    df=df[~df['professional'].isnull()]
    print('labeled:',len(df))
    df['professional']=df.professional.astype(int)
    df.rename(columns={'tweet-id':'tweet_id','professional':coder},inplace=True)
    return df[['tweet_id','text',coder,'url']].set_index('tweet_id',drop=True)

# In[4]:

servane1= load_data('data/servane-1-500.xlsx','Servane')
lucas1 = load_data('data/lucas-1-500.xlsx','Lucas')
jevante1 = load_data('data/jevante-1-500.xlsx','Jevante')
aidan1 = load_data('data/aidan-1-500.xlsx','Aidan')

servane2 = load_data('data/servane-500-1000.xlsx','Servane')
lucas2 = load_data('data/lucas-500-1000.xlsx','Lucas')
jevante2 = load_data('data/jevante-500-1000.xlsx','Jevante')
aidan2 = load_data('data/aidan-500-1000.xlsx','Aidan')

servane3 = load_data('data/servane-1000-1500.xlsx','Servane')
lucas3 = load_data1('data/lucas-1000-1500.csv','Lucas')
jevante3 = load_data('data/jevante-1000-1500.xlsx','Jevante')
aidan3 = load_data('data/aidan-1000-1500.xlsx','Aidan')

servane4 = load_data('data/servane-1500-2000.xlsx','Servane')
lucas4 = load_data1('data/lucas-1500-2000.csv','Lucas')
jevante4 = load_data('data/jevante-1500-2000.xlsx','Jevante')
aidan4 = load_data('data/aidan-1500-2000.xlsx','Aidan')

servane5 = load_data('data/servane-2000-2500.xlsx','Servane')
lucas5 = load_data1('data/lucas-2000-2500.csv','Lucas')
jevante5 = load_data('data/jevante-2000-2500.xlsx','Jevante')
aidan5 = load_data('data/aidan-2000-2500.xlsx','Aidan')

# servane6 = load_data1('data/servane-2500-3000.csv','Servane')
# lucas6 = load_data1('data/lucas-25000-3000.csv','Lucas')
# jevante6 = load_data('data/jevante-2500-3000.xlsx','Jevante')
# aidan6 = load_data('data/aidan-2500-3000.xlsx','Aidan')


# In[5]:
# This is for 1-500.xlsx
print(servane1.columns,lucas1.columns,jevante1.columns, aidan1.columns)
servane1=servane1.reset_index(drop=False)
aidan1=aidan1.reset_index(drop=False)
jevante1=jevante1.reset_index(drop=False)
lucas1=lucas1.reset_index(drop=False)
combined1 = servane1.merge(aidan1,on=['text','url','tweet_id']).merge(lucas1, on=['text','url',"tweet_id"]).merge(jevante1, on=['text','url',"tweet_id"])
print(combined1.columns)
combined1.set_index('tweet_id', inplace=True)
combined1 = combined1[['Servane','Aidan','Lucas','Jevante', 'text','url']]
combined1.shape

# # This is for 500-1000.xlsx
print(servane2.columns,lucas2.columns,jevante2.columns, aidan2.columns)
servane2=servane2.reset_index(drop=False)
aidan2=aidan2.reset_index(drop=False)
jevante2=jevante2.reset_index(drop=False)
lucas2=lucas2.reset_index(drop=False)
combined2 = servane2.merge(aidan2,on=['text','url','tweet_id']).merge(lucas2, on=['text','url',"tweet_id"]).merge(jevante2, on=['text','url',"tweet_id"])
print(combined2.columns)
combined2.set_index('tweet_id', inplace=True)
combined2 = combined2[['Servane','Aidan','Lucas','Jevante', 'text','url']]
combined2.shape

# # This is for 1000-1500.xlsx
print(servane3.columns,lucas3.columns,jevante3.columns, aidan3.columns)
servane3=servane3.reset_index(drop=False)
aidan3=aidan3.reset_index(drop=False)
jevante3=jevante3.reset_index(drop=False)
lucas3=lucas3.reset_index(drop=False)
combined3 = servane3.merge(aidan3,on=['text','url','tweet_id']).merge(lucas3, on=['text','url',"tweet_id"]).merge(jevante3, on=['text','url',"tweet_id"])
print(combined3.columns)
combined3.set_index('tweet_id', inplace=True)
combined3 = combined3[['Servane','Aidan','Lucas','Jevante', 'text','url']]
combined3.shape

# # This is for 1500-2000.xlsx
print(servane4.columns,lucas4.columns,jevante4.columns, aidan4.columns)
servane4=servane4.reset_index(drop=False)
aidan4=aidan4.reset_index(drop=False)
jevante4=jevante4.reset_index(drop=False)
lucas4=lucas4.reset_index(drop=False)
combined4 = servane4.merge(aidan4,on=['text','url','tweet_id']).merge(lucas4, on=['text','url',"tweet_id"]).merge(jevante4, on=['text','url',"tweet_id"])
print(combined4.columns)
combined4.set_index('tweet_id', inplace=True)
combined4 = combined4[['Servane','Aidan','Lucas','Jevante', 'text','url']]
combined4.shape

# # This is for 2000-2500.xlsx
print(servane5.columns,lucas5.columns,jevante5.columns, aidan5.columns)
servane5=servane5.reset_index(drop=False)
aidan5=aidan5.reset_index(drop=False)
jevante5=jevante5.reset_index(drop=False)
lucas5=lucas5.reset_index(drop=False)
combined5 = servane5.merge(aidan5,on=['text','url','tweet_id']).merge(lucas5, on=['text','url',"tweet_id"]).merge(jevante5, on=['text','url',"tweet_id"])

combined1 = pd.concat([combined1,combined2,combined3,combined4,combined5])
combined1.shape
print(combined5.columns)
combined5.set_index('tweet_id', inplace=True)
combined5 = combined5[['Servane','Aidan','Lucas','Jevante', 'text','url']]
combined5.shape

# # This is for 2500-3000.xlsx
# print(servane6.columns,lucas6.columns,jevante6.columns, aidan6.columns)
# servane6=servane6.reset_index(drop=False)
# aidan6=aidan6.reset_index(drop=False)
# jevante6=jevante6.reset_index(drop=False)
# lucas6=lucas6.reset_index(drop=False)
# combined6 = servane6.merge(aidan6,on=['text','url','tweet_id']).merge(lucas6, on=['text','url',"tweet_id"]).merge(jevante6, on=['text','url',"tweet_id"])
# print(combined6.columns)
# combined6.set_index('tweet_id', inplace=True)
# combined6 = combined6[['Servane','Aidan','Lucas','Jevante', 'text','url']]
# combined6.shape








# In[54]:

combined1.head(3)
combined2.head(3)
combined3.head(3)
combined4.head(3)
combined5.head(3)
# combined6.head(3)


# In[7]:

# matched = combined[combined.apply(lambda row: (row.Servane+row.Aidan+row.Lucas+row.Jevante)==0 or
#                 (row.Servane+row.Aidan+row.Lucas+row.Jevante)==4, axis=1)]
# len(matched), len(matched)/len(combined)


# In[8]:

data=combined1[['text']]
data['label']=combined1.apply(lambda row: 0 if (row.Servane+row.Aidan+row.Lucas+row.Jevante)<2 else 1, axis=1)
data.head()


# In[11]:

data['char_count'] = data.apply(lambda row: len(row.text), axis=1)
data['word_count'] = data.apply(lambda row: len(row.text.split()), axis=1)

data['noun_count'] = data.apply(lambda row: get_noun_count(row.text), axis=1)
data['adj_count'] = data.apply(lambda row: get_adjective_count(row.text), axis=1)
data['adverb_count'] = data.apply(lambda row: get_adverb_count(row.text), axis=1)
data.head()


# In[55]:

for feat in ['char_count','word_count','noun_count','adverb_count','adj_count']:
    cor = stats.spearmanr(data[feat],data.label)
    print('({},label):{:.2f}, p:{:.4f}'.format(feat,cor[0],cor[1]))


# In[56]:

pos = data[data.label==1]
neg = data[data.label==0]
print('Pos:{}, neg:{}'.format(len(pos),len(neg)))
fig,axes = plt.subplots(nrows=2,ncols=2, figsize=(18,10))

sns.kdeplot(pos.noun_count, label='pos.noun_count', ax=axes[0][0])
sns.kdeplot(neg.noun_count, label='neg.noun_count',ax=axes[0][0])

sns.kdeplot(pos.word_count, label='pos.word_count', linestyle='-',ax=axes[0][1])
sns.kdeplot(neg.word_count, label='neg.word_count', linestyle='-',ax=axes[0][1])

sns.kdeplot(pos.char_count, label='pos.char_count', linestyle='-',ax=axes[1][0])
sns.kdeplot(neg.char_count, label='neg.char_count', linestyle='-',ax=axes[1][0])

sns.kdeplot((pos.adverb_count), label='pos.adverb_count',ax=axes[1][1])
sns.kdeplot((neg.adverb_count), label='neg.adverb_count', ax=axes[1][1])

plt.show() # This is the first analyzation


# In[49]:

# servane1= load_data('data/servane-1-500.xlsx','Servane')
# lucas1 = load_data('data/lucas-1-500.xlsx','Lucas')
# jevante1 = load_data('data/jevante-1-500.xlsx','Jevante')
# aidan1 = load_data('data/aidan-1-500.xlsx','Aidan')

# servane1=servane1.reset_index(drop=False)
# aidan1=aidan1.reset_index(drop=False)
# jevante1=jevante1.reset_index(drop=False)
# lucas1=lucas1.reset_index(drop=False)
# combined1 = servane1.merge(aidan1,on=['text','url','tweet_id']).merge(lucas1, on=

plt.figure(figsize=(12,8))
sns.stripplot('word_count', 'char_count', data=pos, color='r', jitter=0.4, label='Pos')
sns.stripplot('word_count', 'char_count', data=neg, color='b', jitter=0.4, label='Neg')

plt.show() # This is the second analyzation

# In[41]:
plt.figure(figsize=(12,8))
sns.scatterplot(pos.char_count,pos.word_count, label='Positive')
sns.scatterplot(neg.char_count,neg.word_count, label='Negative')
plt.show() # This is the third analyzation



# In[17]:

mismatched = combined.drop(matched.index)
print(mismatched.shape)
mismatched.to_excel('mismatched-500-1000.xlsx')
mismatched.head()

mismatched = combined.drop(matched.index)
print(mismatched.shape)
mismatched.to_excel('mismatched-1000-1500.xlsx')
mismatched.head()


X= data[['word_count', 'char_count']]
y=data.label
clf = sm.Logit(endog=y, exog=X).fit(disp = False)
print(clf.summary())

# In[ ]:



