import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

# Definimos la función de Himmelblau
def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Definimos los límites de la búsqueda: -5 <= x, y <= 5
bounds = [(-5, 5), (-5, 5)]

# Ejecutamos el algoritmo de recocido simulado
result = dual_annealing(himmelblau, bounds)

# Mostramos los resultados
print("Punto óptimo (x, y):", result.x)
print("Valor mínimo de la función:", result.fun)

# Graficamos la función de Himmelblau y el punto mínimo encontrado
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.scatter(result.x[0], result.x[1], color='red', label='Mínimo')
plt.title('Función de Himmelblau y el mínimo encontrado')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
