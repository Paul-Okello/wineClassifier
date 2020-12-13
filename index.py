import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score


#Reading the data
df = pd.read_csv("winequality-white.csv", delimiter=";")


#Visual Analysis of the 🍷 

# for label in df.columns[:-1]:
#    #for label in  ["alcohol"]:
#     plt.boxplot([df[df['quality'] == i][label] for i in range(1,11)]) 
#     plt.title(label)
#     plt.xlabel("Quality")
#     plt.ylabel(label)
#     plt.savefig("imgs/"+"white".join(label.split(" ")))
#     plt.show()
    
#Gathering training and testing data 
bins = [0,5.5,7.5,10]
labels = [0,1,2]
df["quality"] = pd.cut(df["quality"], bins=bins, labels=labels)
df.head()
print(df.head())

x = df[df.columns[:-1]]
y = df["quality"]
sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=.2, random_state=42)

for data in (y_train, y_test):
    print(data.describe())
    
#Classification of the data(K Nearest Neighbour)

n3 = KNeighborsClassifier(n_neighbors=3)
n3.fit(x_train, y_train)
pred_n3 = n3.predict(x_test)
print(classification_report(y_test, pred_n3))
cross_val = cross_val_score(estimator=n3, X=x_train, y=y_train, cv=10)
print(cross_val.mean()*100)

#Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
print(classification_report(y_test, pred_rf))
cross_val = cross_val_score(estimator=rf, X=x_train, y=y_train, cv=10)
print (cross_val.mean()*100)

#Decision Tree Classifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
print(classification_report(y_test, pred_dt))
cross_val = cross_val_score(estimator=dt, X=x_train, y=y_train, cv=10)
print (cross_val.mean()*100)