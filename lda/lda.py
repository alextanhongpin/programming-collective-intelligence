
# latent-dirachalet-allocation
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
import string


doc1 = 'Technology such as JavaScript has been exceedingly popular'
doc2 = 'Why program in JavaScript versus Golang'
doc3 = '30 students skips class today'
doc4 = 'Programming in python is awesome'
doc5 = 'Healthy food is good'

# Compile documents
docs = [doc1, doc2, doc3, doc4, doc5]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Cleaning and preprocessing
doc_clean = [clean(doc).split() for doc in docs] 
print(doc_clean)

# Preparing Document-Term matrix

import gensim
from gensim import corpora

# Create the term dictionary of our corpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(doc_clean)
print(dictionary)

# Converting list of documents (bag of words) into Document Term Matrix using dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print('doc_term_matrix', doc_term_matrix)

# Running lda model
# Creating the object for LDA model using gensim library
lda = gensim.models.ldamodel.LdaModel

# Running and training LDA model on the document term matrix
ldamodel = lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)

# Results
print(ldamodel.print_topics(num_topics=3, num_words=3))

# Infer topic distributions with new documents
# print(ldamodel['javascript is good'])
# ldamodel.save('lda.pkl')

bow = dictionary.doc2bow(['javascript', 'cool'])
print(ldamodel[bow])

print(ldamodel.show_topic(1))