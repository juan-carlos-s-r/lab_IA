import numpy as np
import random


tablero=np.zeros((4,4),dtype=str)


def imprimir_tablero():
    return print(f"""
                0   1   2  3
            0   {tablero[0,0]} | {tablero[0,1]} | {tablero[0,2]} | {tablero[0,3]}
            1   {tablero[1,0]} | {tablero[1,1]} | {tablero[1,2]} | {tablero[1,3]}
            2   {tablero[2,0]} | {tablero[2,1]} | {tablero[2,2]} | {tablero[2,3]}
            3   {tablero[3,0]} | {tablero[3,1]} | {tablero[3,2]} | {tablero[3,3]}
          """)
    

def  juego_humano_vs_ia():
    
    if condiciones():
        pass
    else:
        x=int(input("En que fila quieres colocar la 'X': "))
        y=int(input("En que columna quieres colocar la 'X': "))
        if tablero[x,y]=='':
            tablero[x,y]='X'
            imprimir_tablero()
        else: 
            print("Poscion ocupada")
            juego_humano_vs_ia()

        #turno de la ia
        mejor_valor = -float('inf')
        mejor_jugada = None
        for x in range(4):
            for y in range(4):
                if tablero[x,y]=='':
                    tablero[x, y] = 'O'
                    valor = minimax(tablero, 0, False, -float('inf'), float('inf'))
                    tablero[x, y] = ''
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_jugada = (x, y)
        
        if mejor_jugada:
            tablero[mejor_jugada[0], mejor_jugada[1]] = 'O'
            imprimir_tablero()
        juego_humano_vs_ia()
        
    
def juego_humano_vs_humano():
    
    if condiciones():
        pass
    else:
        
        x=int(input("En que fila quieres colocar la 'X': "))
        y=int(input("En que columna quieres colocar la 'X': "))
        if tablero[x,y]=='':
            tablero[x,y]='X'
            imprimir_tablero()
        else: 
            print("Poscion ocupada")
            juego_humano_vs_humano()
        x=int(input("En que fila quieres colocar la 'O': "))
        y=int(input("En que columna quieres colocar la 'O': "))
        if tablero[x,y]=='':
            tablero[x,y]='O'
            imprimir_tablero()
        else:
            print("Poscion ocupada")
            juego_humano_vs_humano()
        
        juego_humano_vs_humano()

def juego_ia_vs_ia():
    turno = True  # True para IA con 'O', False para IA con 'X'
    
    while True:
        imprimir_tablero()
        
        if turno:
            # Turno de la IA 'O'
            mejor_valor = -float('inf')
            mejor_jugada = None
            for i in range(4):
                for j in range(4):
                    if tablero[i, j] == '':
                        tablero[i, j] = 'O'
                        valor = minimax(tablero, 0, False, -float('inf'), float('inf'))
                        tablero[i, j] = ''
                        if valor > mejor_valor:
                            mejor_valor = valor
                            mejor_jugada = (i, j)
            
            if mejor_jugada:
                tablero[mejor_jugada[0], mejor_jugada[1]] = 'O'
                imprimir_tablero()
            if condiciones():
                break
            turno = False  # Cambiar turno a IA 'X'

        else:
            # Turno de la IA 'X'
            mejor_valor = float('inf')
            mejor_jugada = None
            for i in range(4):
                for j in range(4):
                    if tablero[i, j] == '':
                        tablero[i, j] = 'X'
                        valor = minimax(tablero, 0, True, -float('inf'), float('inf'))
                        tablero[i, j] = ''
                        if valor < mejor_valor:
                            mejor_valor = valor
                            mejor_jugada = (i, j)
            
            if mejor_jugada:
                tablero[mejor_jugada[0], mejor_jugada[1]] = 'X'
                imprimir_tablero()
            if condiciones():
                break
            turno = True  # Cambiar turno a IA 'O'

                

def condiciones():
    if '' not in tablero:
        print("La partida termino en empate")
        return True
    elif np.all(np.diag(tablero)=='X'):
        print("!!Ganaste X¡¡")
        imprimir_tablero()
        return True
    elif np.all(np.diag(tablero)=='O'):
        print("!!Ganaste O¡¡")
        imprimir_tablero()
        return True
    elif np.all(np.diag(np.fliplr(tablero))=='X'):
        print("!!Ganaste X¡¡")
        imprimir_tablero()
        return True
    elif np.all(np.diag(np.fliplr(tablero))=='O'):
        print("!!Ganaste O¡¡")
        imprimir_tablero()
        return True
    else:
        for i in range(4):
            fila=tablero[i, :]
            if np.all(fila=='X'):
                print("!!Ganaste X¡¡")
                imprimir_tablero()
                return True
            elif np.all(fila=='O'):
                print("!!Ganaste O¡¡")
                imprimir_tablero()
                return True
        for j in range(4):
            columna=tablero[:, j]
            if np.all(columna=='X'):
                print("!!Ganaste X¡¡")
                imprimir_tablero()
                return True
            elif np.all(columna=='O'):
                print("!!Ganaste O¡¡")
                imprimir_tablero()
                return True

cache = {}

def minimax(tablero, profundidad, es_max, alpha, beta):
    estado = tablero.tostring()  # Convertir tablero en una cadena para usarla como clave
    if estado in cache:
        return cache[estado]
    
    elif profundidad == 0 or condiciones():  # Evaluar victoria o empate
        evaluacion = evaluar_tablero(tablero)
        cache[estado] = evaluacion
        return evaluacion
    
    elif es_max:
        max_eval = -float('inf')
        for i in range(4):
            for j in range(4):
                if tablero[i, j] == '':
                    tablero[i, j] = 'O'
                    evaluacion = minimax(tablero, profundidad - 1, False, alpha, beta)
                    tablero[i, j] = ''
                    max_eval = max(max_eval, evaluacion)
                    alpha = max(alpha, evaluacion)
                    if beta <= alpha:
                        break
        cache[estado] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(4):
            for j in range(4):
                if tablero[i, j] == '':
                    tablero[i, j] = 'X'
                    evaluacion = minimax(tablero, profundidad - 1, True, alpha, beta)
                    tablero[i, j] = ''
                    min_eval = min(min_eval, evaluacion)
                    beta = min(beta, evaluacion)
                    if beta <= alpha:
                        break
        cache[estado] = min_eval
        return min_eval


def evaluar_tablero(tablero):
    # Revisar filas y columnas
    for i in range(4):
        if np.all(tablero[i, :] == 'X'):
            return -10  # Ganó el jugador humano
        if np.all(tablero[i, :] == 'O'):
            return 10  # Ganó la IA
        if np.all(tablero[:, i] == 'X'):
            return -10
        if np.all(tablero[:, i] == 'O'):
            return 10
    
    # Revisar diagonales
    if np.all(np.diag(tablero) == 'X'):
        return -10
    if np.all(np.diag(tablero) == 'O'):
        return 10
    if np.all(np.diag(np.fliplr(tablero)) == 'X'):
        return -10
    if np.all(np.diag(np.fliplr(tablero)) == 'O'):
        return 10
    
    # Verificar si es empate
    if '' not in tablero:
        return 0
    
    # No hay ganador ni empate
    return 0

def seleccionar_modalidad():
    modalidad = int(input("Selecciona la modalidad: \n1. Humano vs Humano\n2. Humano vs IA\n3. IA vs IA\n"))
    
    if modalidad == 1:
        imprimir_tablero()
        juego_humano_vs_humano()
    elif modalidad == 2:
        imprimir_tablero()
        juego_humano_vs_ia()
    elif modalidad == 3:
        juego_ia_vs_ia()
    else:
        print("Opción no válida")


if __name__=="__main__":
    seleccionar_modalidad()