import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data=pd.read_csv('tsn.csv')
print("first five records\n",data.head())

x=data.iloc[:,:-1]
print("first five train data\n",x.head())

y=data.iloc[:,-1]
print("first five train output\n",y.head())

x=x.copy()

le = LabelEncoder()
x.Outlook = le.fit_transform(x.Outlook)

x.Temperature=le.fit_transform(x.Temperature)

x.Humidity=le.fit_transform(x.Humidity)

x.Wind=le.fit_transform(x.Wind)

print("after encoding train data\n",x.head())

y=y.copy()

y=le.fit_transform(y)

print("after encoding test data\n",y)

x_train, x_test, y_train, y_test=train_test_split(x,y)

classifier=GaussianNB()
classifier.fit(x_train,y_train)

print("Accuracy ", classifier.score(x_test, y_test)*100)