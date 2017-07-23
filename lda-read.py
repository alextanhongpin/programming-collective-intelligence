

import gensim
from gensim.models.ldamodel import LdaModel
lda = LdaModel.load('lda.pkl', mmap='r')
print(lda.print_topics(num_topics=3, num_words=3))

print(lda['javascript'])