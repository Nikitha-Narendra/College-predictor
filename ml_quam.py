import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#read dataset
dataset = pd.read_csv("cleaned_data_20.csv")

#split into test and train data
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1, shuffle = True)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

#cross validate different ML algorithms
for name,model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle = True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(),cv_results.std()))

#train model
model =  DecisionTreeClassifier()
model.fit(X_train,y_train)

#test model
predictions = model.predict(X_test)

#validate model
print(accuracy_score(y_test,predictions))
