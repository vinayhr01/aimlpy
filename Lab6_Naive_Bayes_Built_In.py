'''from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np

dataset = list(csv.reader(open('diabetes2.csv')))
splitratio = 0.7

trainSize = int(len(dataset) * splitratio)

print("Split {0} rows into train={1} and test={2} rows using built in functions".format(len(dataset),trainSize,len(dataset)-trainSize))

X_train = [[float(j) for j in i[:-1]] for i in dataset[1:trainSize]]
print(X_train)
X_test = [[float(j) for j in i[:-1]] for i in dataset[trainSize:]]

y_train = [float(i[-1]) for i in dataset[1:trainSize]]
y_test = [float(i[-1]) for i in dataset[trainSize:]]

model = GaussianNB()
model.fit(X_train, y_train)
cnt = 0
for i in range(len(X_test)): 
   prediction = model.predict(np.array([X_test[i]])) 
   if prediction[0] == y_test[i]:
       cnt += 1
print("Accuracy ", (cnt/len(X_test))*100)
print("Accuracy ", model.score(X_test,y_test)*100)
'''
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

le_outlook = LabelEncoder()
x.Outlook = le_outlook.fit_transform(x.Outlook)

le_Temperature=LabelEncoder()
x.Temperature=le_Temperature.fit_transform(x.Temperature)

le_Humidity=LabelEncoder()
x.Humidity=le_Humidity.fit_transform(x.Humidity)

le_Wind=LabelEncoder()
x.Wind=le_Wind.fit_transform(x.Wind)

print("after encoding train data\n",x.head())

y=y.copy()

le_PlayTennis=LabelEncoder()
y=le_PlayTennis.fit_transform(y)

print("after encoding test data\n",y)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.7,random_state=0)

classifier=GaussianNB()
classifier.fit(x_train,y_train)

print("Accuracy ", classifier.score(x_test, y_test)*100)