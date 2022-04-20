import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("iris.csv")
# print(data["variety"])
predict = "variety"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_test = [[5,2,1,1], [2,7,4,1]]

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, Y)
acc = model.predict(x_test)
print(acc)