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
df.head()
print(df.head())

#Visual Analysis of the 🍷 

for label in df.columns[:-1]:
   #for label in  ["alcohol"]:
    plt.boxplot([df[df['quality'] == i][label] for i in range(1,11)]) 
    plt.title(label)
    plt.xlabel("Quality")
    plt.ylabel(label)
    plt.savefig("imgs/"+"white".join(label.split(" ")))
    plt.show()