import pandas as pd
data = [[5.3, 3.7, 1.5, 0.2, 'Iris-setosa'], [5, 3.3, 1.4, 0.2, 'Iris-setosa'], [7, 3.2, 4.7, 1.4, 'Iris-versicolor'], [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor']]
df = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4', 'Class'])
# print(df['Class'])
df['Class'] = df['Class'].eq('Iris-versicolor').mul(1)
print(df)