import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

results = []
news_content = """
Imagine a scenario where your application is handling a lot of users at the same time with various functionalities. To implement such a scenario smoothly we should have a server that is able to handle all this. If the users keep on increasing we should scale the server and do all the necessary steps for the smooth working of our application. This is time-consuming and costly. Think if there is a way to automate it, that too at low cost.

AWS Lambda does the work for us. AWS Lambda is a new AWS service that helps us to not only scale the most time and server consuming part for us but also carry out error logging, patching and monitoring. All we need to provide is the code. And we pay as per our use. 

 

When does it call?
You can use AWS Lambda to run your code in response to events, like changes to data in an Amazon S3 bucket, an Amazon DynamoDB table, to run your code in response to HTTP requests using Amazon API Gateway, or invoke your code using API calls made using AWS SDKs. With these capabilities, you can use Lambda to easily build data processing triggers for AWS services like Amazon S3 and Amazon DynamoDB process streaming data stored in Amazon Kinesis, or create your own back end that operates at AWS scale, performance, and security.
"""

sentences = nltk.sent_tokenize(news_content)
print(sentences)

vectorizer = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)

sklearn_binary = vectorizer.fit_transform(sentences)
print(sklearn_binary)
print(sklearn_binary.toarray())

for i in sklearn_binary.toarray():
    results.append(i.sum() / float(len(i.nonzero()[0])))

print(results)