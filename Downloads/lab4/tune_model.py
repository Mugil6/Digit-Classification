from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grid Search
params = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1]
}
grid = GridSearchCV(SVC(), params, cv=5)
grid.fit(X_train, y_train)

# Best params and accuracy
print("Best Params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
