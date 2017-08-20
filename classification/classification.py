

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# import pickle
from sklearn.externals import joblib as pickle
import numpy as np

# documents = [open(f) for f in text_files]
# tfidf = TfidfVectorizer().fit_transform(documents)
# # no need to normalize, since Vectorizer will return normalized tf-idf
# pairwise_similarity = tfidf * tfidf.T
vect = TfidfVectorizer(min_df=0, use_idf=True, norm='l2', stop_words='english')
# documents that should be cleaned, no stop words and no symbols, lowercase
documents = [
    'food is good and is not delicious',
    'technology is advancing and need good programmers',
    'programmers are learning new languages',
    'this food is delicous',
    'war is unevitable',
    'stop war',
    'flowers a, flowers b, flowers c',
    'flowers d'
]

tfidf = vect.fit_transform(documents)
print('tfidf:', tfidf.toarray())
#Save vect.vocabulary_
pickle.dump(vect.vocabulary_,open("classification.pkl","wb"))


loaded_vec = TfidfVectorizer(vocabulary=pickle.load(open("classification.pkl", "rb")))
unseen_data = ['technology in 2017 is advancing', 'world war 3', 'food']
unseen_tfifd = loaded_vec.fit_transform(unseen_data)
# unseen_tfifd = loaded_vec.transform(unseen_data)
print('unseen_tfifd:', unseen_tfifd)
print('vect.get_feature_names', loaded_vec.get_feature_names())

# unseen_tfidf = vec.transform(unseen_data)
km = KMeans(n_clusters=4, verbose=0)

kmresult = km.fit(tfidf).predict(unseen_tfifd)
print("result", kmresult)
print('cluster_centers_:', km.cluster_centers_)
print('labels_:', km.labels_)
print('inertia_:', km.inertia_)


## Load it 
# loaded_vec = TfidfVectorizer(vocabulary=pickle.load(open("feature.pkl", "rb")))
# tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))

# results = tfidf.toarray()
# print(results)
# Notes
# Classification by kmeans work, but need to find out how to discover topic
# Use Latent dirchalet allocation to discover different topic
