import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data=pd.read_csv('tsn.csv')

x=data.iloc[:,:-1]

y=data.iloc[:,-1]

x=x.copy()

le = LabelEncoder()
x.Outlook = le.fit_transform(x.Outlook)

x.Temperature=le.fit_transform(x.Temperature)

x.Humidity=le.fit_transform(x.Humidity)

x.Wind=le.fit_transform(x.Wind)

y=y.copy()

y=le.fit_transform(y)

x_train, x_test, y_train, y_test=train_test_split(x,y)

classifier=GaussianNB()
classifier.fit(x_train,y_train)

print("Accuracy ", classifier.score(x_test, y_test)*100)