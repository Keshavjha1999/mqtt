import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier


data = pd.read_csv("iris.csv")
# print(data["variety"])
predict = "variety"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
kf = model_selection.KFold(n_splits=10)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

for train_indices, test_indices in kf.split(X):
    clf.fit(X[train_indices], Y[train_indices])
    print(clf.score(X[test_indices], Y[test_indices]))