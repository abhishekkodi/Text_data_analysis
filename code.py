# Text_data_analysis


# 11111111111111111111111

from __future__ import division
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

# Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
plt.gray();



# 222222222222222222222222222

#Import Modules

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
    

# Importing data

categories = [
    'sci.electronics', 
    'sci.crypt',
    'sci.med',
    'sci.space',
]
    
train_data = load_files('../datasets/20news-bydate-train/',
    categories=categories, encoding='latin-1')
#test_data = load_files('../datasets/20news-bydate-test/',
                       
                       
    #categories=categories, encoding='latin-1')

list(train_data.keys())
print(load_files.__doc__)

# 333333333333333333333333333333333

# Converting data into vectors
vectorizer = TfidfVectorizer(min_df=2)
X_train = vectorizer.fit_transform(train_data.data)
y_train = train_data.target
print("vocabulary")
text_documents, vocabulary = X_train.shape
print(vocabulary)
print("number of documents")
print(text_documents)

#kk = X_train.target
# Number of vectors
#no_of_words = X_train.len()
#print(no_of_words)
classifier = MultinomialNB().fit(X_train, y_train)
print("Training score: {0:.1f}%".format(
    classifier.score(X_train, y_train) * 100))
    
    
# 44444444444444444444444444444444444

def text_size(text, charset='iso-8859-1'):
    return len(text.encode(charset)) * 8 * 1e-6

train_size_mb = sum(text_size(text) for text in train_data.data) 
test_size_mb = sum(text_size(text) for text in train_data.data)

print("Training set size: {0} MB".format(int(train_size_mb)))
print("Testing set size: {0} MB".format(int(test_size_mb)))

vectorizer = TfidfVectorizer(min_df=1)

%time X_train_small = vectorizer.fit_transform(train_data.data)

from sklearn.decomposition import TruncatedSVD

%time X_train_small_pca = TruncatedSVD(n_components=2).fit_transform(X_train_small)

# 55555555555555555555555555555555555555555555555555555555555

# Training Data
from itertools import cycle

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, c in zip(np.unique(y_train), cycle(colors)):
    plt.scatter(X_train_small_pca[y_train == i, 0],
               X_train_small_pca[y_train == i, 1],
               c=c, label=train_data.target_names[i], alpha=0.5)
    
plt.legend(loc='best');

# 66666666666666666666666666666666666666666666666666666666666

# Testing Data
#X_test = vectorizer.transform(test_data.data)
#y_test = test_data.target

#print("The total tested words")
#print(X_train.shape)
#print("Testing score: {0:.1f}%".format(
    #classifier.score(X_test, y_test) * 100))
    
#cross fold validation
from sklearn.cross_validation import ShuffleSplit
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

from sklearn import metrics 
from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_data,Y_data,cv=5,scoring="F1_Macro")

np.sum(scores["accuracy"])
