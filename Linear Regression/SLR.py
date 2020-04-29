import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#with outlier


datasets = pd.read_excel('blood.xlsx')

X = datasets.iloc[:,1].values
y = datasets.iloc[:,-1].values

X = X.reshape(-1, 1)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


y_pred = lin_reg.predict(X)

lin_reg.score(X,y)

plt.scatter(X,y)
plt.plot(X,y_pred, c = 'b')
plt.show()

lin_reg.predict([[22]])
lin_reg.predict([[26]])

lin_reg.coef_
lin_reg.intercept_
#43%

#without outlier
datasets = pd.read_excel('blood.xlsx')

X = datasets.iloc[2:,1].values
y = datasets.iloc[2:,-1].values

X = X.reshape(-1, 1)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


y_pred = lin_reg.predict(X)

lin_reg.predict([[22]])
lin_reg.predict([[26]])

lin_reg.score(X,y)

plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()

lin_reg.coef_
lin_reg.intercept_
#72%

























