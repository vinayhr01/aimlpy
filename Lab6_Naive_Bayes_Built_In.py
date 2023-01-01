from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np

dataset = list(csv.reader(open('diabetes2.csv')))
splitratio = 0.7

trainSize = int(len(dataset) * splitratio)

print("Split {0} rows into train={1} and test={2} rows using built in functions".format(len(dataset),trainSize,len(dataset)-trainSize))

X_train = [[float(j) for j in i[:-1]] for i in dataset[:trainSize]]
X_test = [[float(j) for j in i[:-1]] for i in dataset[trainSize:]]

y_train = [float(i[-1]) for i in dataset[:trainSize]]
y_test = [float(i[-1]) for i in dataset[trainSize:]]

model = GaussianNB()
model.fit(X_train, y_train)
print("Accuracy ", model.score(X_test,y_test)*100)