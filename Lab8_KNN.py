# Python program to demonstrate # KNN classification algorithm # on IRISdataset 
 #Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. 
 #Print both correct and wrong predictions. Java/Python ML library classes can be used for 
 #this problem. 
 
 #import the dataset and library files 
from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 
from sklearn.model_selection import train_test_split 
 
iris_dataset=load_iris() 
 
 #split the data into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"]) 
 
 #train and fit the model 
kn = KNeighborsClassifier(n_neighbors=5) 
kn.fit(X_train, y_train) 
 
for i in range(len(X_test)): 
   prediction = kn.predict(np.array([X_test[i]])) 
   print("\n Actual : {0} {1}, Predicted :{2}{3}".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][ prediction])) 
print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test))) 
