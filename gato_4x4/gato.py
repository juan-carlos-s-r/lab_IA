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
    

def  juego_maquina():
    list_temp=[]
    for index, value in np.ndenumerate(tablero):
        if value=='':
            list_temp.append(index)
    
    random_temp=random.randint(0,len(list_temp)-1)
    index_temp=list_temp[random_temp]

    if condiciones():
        pass      
    else:
        tablero[index_temp]='O'
        imprimir_tablero()
        juego_humano()
        
    
def juego_humano():
    
    if condiciones():
        pass
    else:
        
        x=int(input("En que fila quieres colocar la 'X': "))
        y=int(input("En que columna quieres colocar la 'X': "))
        if tablero[x,y]!='X' and tablero[x,y]!='O':
            tablero[x,y]='X'
            juego_maquina()
        else: 
            print("Poscion ocupada")
            juego_humano()
                

def condiciones():
    if '' not in tablero:
        print("La partida termino en empate")
        return True
    elif np.all(np.diag(tablero)=='X'):
        print("!!Ganaste¡¡")
        imprimir_tablero()
        return True
    elif np.all(np.diag(tablero)=='O'):
        print("Perdiste :(")
        imprimir_tablero()
        return True
    elif np.all(np.diag(np.fliplr(tablero))=='X'):
        print("!!Ganaste¡¡")
        imprimir_tablero()
        return True
    elif np.all(np.diag(np.fliplr(tablero))=='O'):
        print("Perdiste :(")
        imprimir_tablero()
        return True
    else:
        for i in range(4):
            fila=tablero[i, :]
            if np.all(fila=='X'):
                print("!!Ganaste¡¡")
                imprimir_tablero()
                return True
            elif np.all(fila=='O'):
                print("Perdiste :(")
                imprimir_tablero()
                return True
        for j in range(4):
            columna=tablero[:, j]
            if np.all(columna=='X'):
                print("!!Ganaste¡¡")
                imprimir_tablero()
                return True
            elif np.all(columna=='O'):
                print("Perdiste :(")
                imprimir_tablero()
                return True

if __name__=="__main__":
    imprimir_tablero()
    juego_humano()