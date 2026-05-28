# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize dataset, normalize features, and set weights (θ) to zero.
2.Compute predicted probabilities using the sigmoid function.
3.Update weights iteratively using gradient descent to minimize error. 
4.Predict classes using a threshold (0.5) and evaluate accuracy.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: HITHESH RAJ R K
RegisterNumber:  212225040129
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])

plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")

plt.show()
```

## Output:
<img width="997" height="732" alt="image" src="https://github.com/user-attachments/assets/84d80f6c-2947-43ce-9ac5-50dcdcd52836" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
