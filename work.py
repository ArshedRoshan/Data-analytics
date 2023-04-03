import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data.csv")
print(df)
x = df.iloc[:,:5]
print('xx',x)
y = df.iloc[:, -1]
print('yy',y)



# split data into tarining and testing sets
np.random.seed(123)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred,average='weighted')
recall  = recall_score(y_test,y_pred,average='weighted')
f1 = f1_score(y_test,y_pred,average='weighted')
print('accuracy',accuracy)
print('precision',precision)
print('recall',recall)
print('f1_score',f1)


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

with open('evaluation.csv','w') as file:
    file.write('Accuracy: {:.2f}\n'.format(accuracy))
    file.write('Precision: {:.2f}\n'.format(precision))
    file.write('Recall: {:.2f}\n'.format(recall))
    file.write('F1 score: {:.2f}\n'.format(f1))
    







