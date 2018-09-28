#!/usr/bin/env python
# coding: utf-8

# # Tokenization

# In[1]:


import nltk
text = "This is Andrew's text, isn't it?"


# In[2]:


tokenizer = nltk.tokenize.WhitespaceTokenizer()
tokenizer.tokenize(text)


# In[3]:


tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer.tokenize(text)


# In[4]:


tokenizer = nltk.tokenize.WordPunctTokenizer()
tokenizer.tokenize(text)


# # Stemming (further in the video)

# In[5]:


import nltk
text = "feet wolves cats talked"
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)


# In[6]:


stemmer = nltk.stem.PorterStemmer()
" ".join(stemmer.stem(token) for token in tokens)


# In[7]:


stemmer = nltk.stem.WordNetLemmatizer()
" ".join(stemmer.lemmatize(token) for token in tokens)

