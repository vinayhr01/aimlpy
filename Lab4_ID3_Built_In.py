from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("tsn.csv")

x = data.iloc[:,:-1]

y = data.iloc[:,-1]

x = x.copy()

y = y.copy()

le = LabelEncoder()
x.Outlook = le.fit_transform(x.Outlook)

x.Temperature=le.fit_transform(x.Temperature)

x.Humidity=le.fit_transform(x.Humidity)

x.Wind=le.fit_transform(x.Wind)

y=le.fit_transform(y)

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0)

dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train, y_train)
f_tree = export_text(dtree,feature_names=['Outlook','Temperature','Humidity','Wind'],spacing=10)
print(f_tree)
print("Accuracy is", dtree.score(x_test, y_test)*100)
#plt.figure(figsize=(14,7))
plot_tree(dtree,feature_names=['Outlook','Temperature','Humidity','Wind'],class_names=['Yes','No'],impurity=False,rounded=True,filled=True)
plt.show()
