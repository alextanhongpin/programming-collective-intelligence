import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import sklearn
import csv

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

def preprocessing(text):
    # 
    # text = text.decode('utf8')

    # Tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # Remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # Remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # Lower capitalization
    tokens = [word.lower() for word in tokens]

    # Lemmatize
    lemma = WordNetLemmatizer()

    tokens = [lemma.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


sms = open('SMSSpamCollection')

sms_data = []
sms_labels = []
csv_reader = csv.reader(sms, delimiter='\t')
for line in csv_reader:
    # Adding the sms id
    sms_labels.append(line[0])
    sms_data.append(preprocessing(line[1]))
sms.close()


# 0.7 of the data for training
trainset_size = int(round(len(sms_data) * 0.7))
print('The trainset has {} data'.format(trainset_size))

x_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])
y_train = np.array([el for el in sms_labels[0:trainset_size]])

x_test = np.array([''.join(el) for el in sms_data[trainset_size+1:len(sms_data)]])
y_test = np.array([el for el in sms_labels[trainset_size+1:len(sms_data)]])

print('x_train', x_train, 'x_train')
print('y_train', y_train, 'y_train')

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words='english', strip_accents='unicode', norm='l2')

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

clf = MultinomialNB().fit(X_train, y_train)
y_nb_predicted = clf.predict(X_test)
print(y_nb_predicted)

cm = confusion_matrix(y_test, y_nb_predicted)
print('confusion matrix', cm)

print('classification_report', classification_report(y_test, y_nb_predicted))


print(clf.predict(vectorizer.transform(['Hey, how are you doing?'])))
print(clf.predict(vectorizer.transform(['Call 014 32233 and buy XXXX for $100000?'])))

# Print top features
feature_names = vectorizer.get_feature_names()
coefs = clf.coef_
intercept = clf.intercept_
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n = 10
top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print('{} {} \t {} {}'.format(coef_1, fn_1, coef_2, fn_2))