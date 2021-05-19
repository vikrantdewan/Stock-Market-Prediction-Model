import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data = pd.read_csv('Full_Data.csv', encoding = "ISO-8859-1")
data.head(1)

train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']

# Removing punctuations
slicedData= train.iloc[:,2:27]
slicedData.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
slicedData.columns= new_Index
slicedData.head(5)

# Convertng headlines to lower case
for index in new_Index:
    slicedData[index]=slicedData[index].str.lower()
slicedData.head(1)

headlines = []
for row in range(0,len(slicedData.index)):
    headlines.append(' '.join(str(x) for x in slicedData.iloc[row,0:25]))

headlines[0]

basicvectorizer = CountVectorizer(ngram_range=(1,1))
basictrain = basicvectorizer.fit_transform(headlines)


basicmodel = GaussianNB()
basicmodel = basicmodel.fit(basictrain.toarray(), train["Label"])

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest.toarray())

predictions

pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])

print(basictrain.shape)
print(pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"]))
print (classification_report(test["Label"], predictions))
print (accuracy_score(test["Label"], predictions))


X = ['1 gram model']
Y = [0.82]
plt.ylim(0,1)
plt.bar(X, Y)
plt.show()