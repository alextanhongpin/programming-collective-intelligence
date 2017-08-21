from sklearn import tree
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import graphviz

df = pd.read_csv('data.csv')

# print(df.corr())

# print(df.head())

label_encoders = {}
# For each column, process it to numerical data
for i in range(1, len(df.columns)):
    label_encoders[i] = LabelEncoder()
    df.ix[:, i] = label_encoders[i].fit_transform(df.ix[:, i].tolist())

# Select all but the last columns
X = df.ix[:,1:-1]

# Select the last column
y = df.ix[:,-1:]

feature_names = df.columns.tolist()[:-1]
class_names = ['Cat', 'Dog']


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

plays_fetch = label_encoders[1].transform(['No'])
is_grumpy = label_encoders[2].transform(['Yes'])
favorite_food = label_encoders[3].transform(['Cat food'])

print(label_encoders[4].inverse_transform(clf.predict([[plays_fetch[0], is_grumpy[0], favorite_food[0]]])))



dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=feature_names,  
                         class_names=class_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data)
graph.render('out')