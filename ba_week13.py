# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=('headers','footers','quotes'))

train_set=twenty_train.data
ytr=twenty_train.target

twenty_valid=fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=('headers','footers','quotes'))
valid_set=twenty_valid.data
yval=twenty_valid.target

import nltk
nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize

doc=train_set[0]

sents=sent_tokenize(doc)
sents

words=word_tokenize(doc)

preprocess_doc = [w for w in words if w.isalnum()]
preprocess_doc = [w for w in words if w.isalpha()]
preprocess_doc = [w.lower() for w in preprocess_doc]

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords=stopwords.words('english')

preprocess_doc = [w for w in preprocess_doc if not w in stopwords]

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

porter_stemmer = PorterStemmer()

porter_stemmer.stem('maximum')
porter_stemmer.stem('presuambly')
porter_stemmer.stem('multiply')
porter_stemmer.stem('owed')
porter_stemmer.stem('crying')


lancaster_stemmer = LancasterStemmer()
lancaster_stemmer.stem('maximum')
lancaster_stemmer.stem('presumably')
lancaster_stemmer.stem('multiply')
lancaster_stemmer.stem('owed')
lancaster_stemmer.stem('crying')

snowball_stemmer = SnowballStemmer('english')
snowball_stemmer.stem('maximum')
snowball_stemmer.stem('presumably')
snowball_stemmer.stem('multiply')
snowball_stemmer.stem('owed')
snowball_stemmer.stem('crying')


nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer=WordNetLemmatizer()

wordnet_lemmatizer.lemmatize('dog')
wordnet_lemmatizer.lemmatize('churches')
wordnet_lemmatizer.lemmatize('hardrock')
wordnet_lemmatizer.lemmatize('are')
wordnet_lemmatizer.lemmatize('are', pos='v')

nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

from nltk import pos_tag

pos_tag(words)
pos_tag(words, tagset='universal')

preprocess_train=[]
for doc in train_set:
    words=word_tokenize(doc)
    
    preprocess_doc = [w for w in words if w.isalpha()]
    preprocess_doc = [w.lower() for w in preprocess_doc]
    preprocess_doc = [w for w in preprocess_doc if not w in stopwords]
    preprocess_doc = [porter_stemmer.stem(w) for w in preprocess_doc]

    preprocess_train.append(''.join(preprocess_doc))
    

preprocess_valid=[]
for doc in valid_set:
    words=word_tokenize(doc)
    
    preprocess_doc = [w for w in words if w.isalpha()]
    preprocess_doc = [w.lower() for w in preprocess_doc]
    preprocess_doc = [w for w in preprocess_doc if not w in stopwords]
    preprocess_doc = [porter_stemmer.stem(w) for w in preprocess_doc]

    preprocess_train.append(''.join(preprocess_doc))
    
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(min_df=1)
vectorizer.fit(preprocess_train)

Xtr=vectorizer.transform(preprocess_train)
Xtr=Xtr.toarray()

features=vectorizer.get_feature_names()

Xval=vectorizer.transform(preprocess_valid)

bigram_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3)
bigram_vectorizer.fit(preprocess_train)


bigram_vectorizer.get_feature_names()

analyzer=bigram_vectorizer.build_analyzer()
analyzer(preprocess_train[0])

tokenizer=vectorizer.build_tokenizer()
tokenizer(preprocess_train[0])
    
from sklearn.feature_extraction.text import TfidfTransformer

transformer=TfidfTransformer(sublinear_tf=True)
transformer.fit(Xtr)
transformer.idf_
len(transformer.idf_)

tfidf_train=transformer.transform(Xtr)
tfidf_valid=transformer.transform(Xval)    
tfidf_train=tfidf_train.toarray()    


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(min_df=10, sublinear_tf=True)

tfidf_vectorizer.fit(preprocess_train)

tfidf_train2=tfidf_vectorizer.transform(preprocess_train)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

fs=SelectKBest(mutual_info_classif, k=500)
fs.fit(tfidf_train,ytr)

fs.scores_

Xtr_reduce = fs.transform(tfidf_train)
Xval_reduce = fs.transform(tfidf_valid)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()
clf.fit(Xtr_reduce, ytr)

clf.score(Xval_reduce,yval)
clf.score(Xtr_reduce, ytr)
