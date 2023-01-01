from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np

dataset = list(csv.reader(open('tsvnaive.csv')))
splitratio = 0.9

trainSize = int(len(dataset) * splitratio)

X_train = [[int(j) for j in i[:-1]] for i in dataset[:trainSize]]
X_test = [[int(j) for j in i[:-1]] for i in dataset[trainSize:]]

y_train = [int(i[-1]) for i in dataset[:trainSize]]
y_test = [int(i[-1]) for i in dataset[trainSize:]]

model = GaussianNB()
model.fit(X_train, y_train)
print("Accuracy ", model.score(X_test,y_test)*100)