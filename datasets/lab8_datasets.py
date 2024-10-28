import pandas as pd

# Define el nombre del archivo
filename = 'C:/Users/juanc/Downloads/bezdekIris.data'

# Define los nombres de las columnas según el dataset Iris
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Carga el dataset en un DataFrame
df = pd.read_csv(filename, header=None, names=column_names)

# Muestra el DataFrame
print(df)