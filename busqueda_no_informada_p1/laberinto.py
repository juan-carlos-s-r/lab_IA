import numpy as np

maze=[
    [1,0,1,1,1],
    [1,0,0,0,1],
    [1,1,1,0,1],
    [1,0,0,0,0],
    [1,1,1,1,1]
]
start=[0,1]
end=[3,4]
nodos_frontera=[]
nodos_visitados=[]
nodos_frontera.append(start)



def moverse_izquierda(nodo_actual):
    posicion=nodo_actual.copy()
    if posicion[-1]!=0:
        posicion[-1]-=1
        return posicion
    else:
        return posicion


def moverse_abajo(nodo_actual):
    posicion=nodo_actual.copy()
    if posicion[0]!=len(maze)-1:
        posicion[0]+=1
        return posicion
    else:
        return posicion

def moverse_derecha(nodo_actual):
    posicion=nodo_actual.copy()
    if posicion[-1]!=len(maze)-1:
        posicion[-1]+=1
        return posicion
    else:
        return posicion

def moverse_arriba(nodo_actual):
    posicion=nodo_actual.copy()
    if posicion[0]!=0:
        posicion[0]-=1
        return posicion
    else:
        return posicion

movimientos=[moverse_derecha, moverse_abajo, moverse_izquierda, moverse_arriba]



while True:
    
    nodo_actual=nodos_frontera.pop()

    if nodo_actual==end:
        print(nodos_visitados)
        break

    nodos_visitados.append(nodo_actual)

    for i in movimientos:
        nodo_hijo=i(nodo_actual)
        
        if nodo_hijo not in nodos_frontera and nodo_hijo not in nodos_visitados:
            if maze[nodo_hijo[0]][nodo_hijo[1]]==0:
                nodos_frontera.append(nodo_hijo)
                
        