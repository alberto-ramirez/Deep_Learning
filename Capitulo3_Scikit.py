import pandas as pd
from sklearn.datasets import load_breast_cancer  ## Dataset predefinido de la libreria
from sklearn.model_selection import train_test_split ## Para dividir los datos
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['tipo'] = dataset.target[df.index]
X = df.iloc[:,:-1]
y = df['tipo'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
reg = LogisticRegression(max_iter=10000)
reg.fit(X_train, y_train)

print("Exactitud {:.2f}".format(accuracy_score(y_test, reg.predict(X_test))))