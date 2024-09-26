nodo_inicial=[1,2,4,3]
solucion=[1,2,3,4]
nodos_frontera=[]
nodos_visitados=[]
nodos_frontera.append(nodo_inicial)


def mover_izquierda(nodo_actual):
    lista_temp=nodo_actual.copy()
    lista_temp[0],lista_temp[1]=lista_temp[1],lista_temp[0]
    return lista_temp

def mover_centro(nodo_actual):
    lista_temp=nodo_actual.copy()
    lista_temp[1],lista_temp[2]=lista_temp[2],lista_temp[1]
    return lista_temp

def mover_derecha(nodo_actual):
    lista_temp=nodo_actual.copy()
    lista_temp[2],lista_temp[3]=lista_temp[3],lista_temp[2]
    return lista_temp

movimientos=[mover_derecha, mover_centro, mover_izquierda]

while True:

    nodo_actual=nodos_frontera.pop(0)

    if nodo_actual==solucion:
        print(nodos_visitados)
        break

    nodos_visitados.append(nodo_actual)

    for i in movimientos:
        nodo_hijo=i(nodo_actual)
        
        if nodo_hijo not in nodos_frontera and nodo_hijo not in nodos_visitados:
            nodos_frontera.append(nodo_hijo)