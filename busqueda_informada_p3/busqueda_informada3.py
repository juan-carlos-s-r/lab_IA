import math
import random

def himmelblau(x,y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def simulated_annealing():
    t=10000
    t_MIN=1e-3
    v_enfriamiento=100
    x_actual= random.uniform(5, -5)
    y_actual= random.uniform(5, -5)
    
    while t > t_MIN:
        for i in range(v_enfriamiento):
            x_nueva= x_actual +  random.uniform(-0.1, 0.1)
            y_nueva= y_actual +  random.uniform(-0.1, 0.1)
            if himmelblau(x_nueva, y_nueva) < himmelblau(x_actual, y_actual):
                x_actual, y_actual= x_nueva, y_nueva
            else:
                if random.random() < math.exp(-(himmelblau(x_nueva, y_nueva)- himmelblau(x_actual, y_actual))/t):
                    x_actual, y_actual= x_nueva, y_nueva
        
        t-=0.005
    return x_actual, y_actual, himmelblau(x_actual, y_actual)

mejor_x, mejor_y, resultado_min= simulated_annealing()

print(f"Mejor x: {mejor_x} \nMejor y: {mejor_y} \nResultado minimo: {resultado_min}")
    

