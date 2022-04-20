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

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.4)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)