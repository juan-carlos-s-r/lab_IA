# Definir la función de Himmelblau
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Establecer el rango de búsqueda para x e y
x_min, x_max = -5, 5
y_min, y_max = -5, 5

# Establecer la resolución de la cuadrícula
step = 0.01  # Cuanto más pequeño, más preciso

# Inicializar variables para almacenar el mínimo
min_x, min_y = None, None
min_value = float('inf')

# Búsqueda por fuerza bruta en la cuadrícula
x = x_min
while x <= x_max:
    y = y_min
    while y <= y_max:
        value = himmelblau(x, y)
        if value < min_value:
            min_value = value
            min_x, min_y = x, y
        y += step
    x += step

# Mostrar los resultados
print(f"Los valores de (x, y) que minimizan la función son: ({min_x}, {min_y})")
print(f"Valor mínimo de la función: {min_value}")
