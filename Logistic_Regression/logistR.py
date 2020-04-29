import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.arange(-10,10,0.01)

sig = 1 / (1 + np.power(np.e,-X))

sig_1 = np.power(np.e,-X) / 1 / (1 + np.power(np.e,-X))


plt.plot(X,sig)
plt.show()




plt.plot(X,sig_1)
plt.show()


lop = 3*X + 7
plt.plot(X,lop)
plt.show()

sig_line =  1 / (1 + np.power(np.e,-lop))
plt.plot(X,sig_line)
plt.show()

sig_line1 = np.power(np.e,-lop) / 1 / (1 + np.power(np.e,-lop))
plt.plot(X,sig_line1)
plt.show()

from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()


X =datasets.data
y =datasets.target

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)




log_reg.score(X,y)
y_pred = log_reg.predict(X)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)


































