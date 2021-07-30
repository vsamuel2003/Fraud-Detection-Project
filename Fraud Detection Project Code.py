import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("fraud.csv")

le = LabelEncoder()
payment_method = le.fit_transform(list(data['paymentMethod']))
data['paymentMethod'] = payment_method

X = data.drop(columns = 'label')
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=17)

log = LogisticRegression()
log.fit(X_train, y_train)
acc = log.score(X_test, y_test)

predictions = log.predict(X_test)
matrix = confusion_matrix(y_test, predictions)

print(f' Accuracy is: {acc}')
print(f' Confusion Matrix is: {matrix}')