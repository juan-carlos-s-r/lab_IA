import numpy as np
from collections import Counter
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer

# Funciones utilitarias
def split_data(X, y, test_size=0.3):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    n_test = int(len(X) * test_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def accuracy(y_true, y_pred):
    """Calcula la exactitud (accuracy)."""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            matrix[i, j] = np.sum((y_true == actual) & (y_pred == predicted))
    return matrix

# Clasificador Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        """Ajusta el modelo calculando probabilidades por clase y promedio/desviación estándar por característica."""
        self.classes = np.unique(y)
        self.means = {c: X[y == c].mean(axis=0) for c in self.classes}
        self.stds = {c: X[y == c].std(axis=0) for c in self.classes}
        self.priors = {c: np.mean(y == c) for c in self.classes}

    def predict(self, X):
        """Predice las clases de los datos X."""
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                likelihood = np.sum(
                    -0.5 * np.log(2 * np.pi * self.stds[c] ** 2)
                    - ((x - self.means[c]) ** 2) / (2 * self.stds[c] ** 2)
                )
                class_probs[c] = np.log(self.priors[c]) + likelihood
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# Clasificador KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Guarda los datos de entrenamiento."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predice las clases de los datos X."""
        predictions = []
        for x in X:
            distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
            neighbors = np.argsort(distances)[:self.k]
            neighbor_labels = self.y_train[neighbors]
            predictions.append(Counter(neighbor_labels).most_common(1)[0][0])
        return np.array(predictions)

# Validaciones
def hold_out(X, y, classifier):
    """Validación Hold-Out."""
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

def k_fold_cross_validation(X, y, classifier, k=10):
    """Validación k-fold."""
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    accuracies = []
    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy(y_test, y_pred))
    return np.mean(accuracies)

def leave_one_out(X, y, classifier):
    """Validación Leave-One-Out."""
    n_samples = len(X)
    accuracies = []
    for i in range(n_samples):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(y_pred[0] == y_test)

    return np.mean(accuracies)

#------------------------iris plant-------------------------------------
print("\n ---------------------------------- iris dataset-----------------------------------")
# Define el nombre del archivo
filename = 'C:/Users/juanc/Downloads/bezdekIris.data'

# Define los nombres de las columnas según el dataset Iris
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Carga el dataset en un DataFrame
df = pd.read_csv(filename, header=None, names=column_names)

# Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
y = df.iloc[:, -1].values   # Última columna

# Evaluación
print("\n--- Naive Bayes ---")
nb = NaiveBayes()
acc, cm = hold_out(X, y, nb)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, nb):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, nb):.4f}")

print("\n--- KNN ---")
knn = KNN(k=5)
acc, cm = hold_out(X, y, knn)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, knn):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, knn):.4f}")

#------------------------------wine dataset---------------------------------------------------
print("\n ---------------------------------- wine dataset-----------------------------------")

data = load_wine()
df_wine = pd.DataFrame(data.data, columns=data.feature_names)
df_wine['class'] = data.target

# Carga el dataset en un DataFrame
df = pd.read_csv(filename, header=None, names=column_names)

# Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
y = df.iloc[:, -1].values   # Última columna

# Evaluación
print("\n--- Naive Bayes ---")
nb = NaiveBayes()
acc, cm = hold_out(X, y, nb)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, nb):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, nb):.4f}")

print("\n--- KNN ---")
knn = KNN(k=5)
acc, cm = hold_out(X, y, knn)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, knn):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, knn):.4f}")

#------------------------------breast cancer dataset---------------------------------------------------
print("\n ---------------------------------- breast cancer dataset-----------------------------------")


data = load_breast_cancer()
df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
df_cancer['class'] = data.target

# Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
y = df.iloc[:, -1].values   # Última columna

# Evaluación
print("\n--- Naive Bayes ---")
nb = NaiveBayes()
acc, cm = hold_out(X, y, nb)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, nb):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, nb):.4f}")

print("\n--- KNN ---")
knn = KNN(k=5)
acc, cm = hold_out(X, y, knn)
print(f"Hold-Out Accuracy: {acc:.4f}")
print(f"Hold-Out Confusion Matrix:\n{cm}")

print(f"10-Fold Cross-Validation Accuracy: {k_fold_cross_validation(X, y, knn):.4f}")
print(f"Leave-One-Out Accuracy: {leave_one_out(X, y, knn):.4f}")
