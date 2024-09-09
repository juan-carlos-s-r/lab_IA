import random

valores_min={'x':None,'y':None,'valor':None,}
cont=-1

while True:
    x_temp=random.uniform(-4.5,4.5)
    y_temp=random.uniform(-4.5,4.5)
    valor_temp=None
    cont+=1

    valor_temp=(1.5 - x_temp + x_temp*y_temp)**2 + (2.25 - x_temp + x_temp*y_temp**2)**2 + (2.625 - x_temp + x_temp*y_temp**3)**2
    
    if valores_min['valor']==None or valores_min['valor']>valor_temp:
        valores_min['x']=x_temp
        valores_min['y']=y_temp
        valores_min['valor']=valor_temp 
        print(f"| valor min de x: {valores_min['x']:.3f} \t| valor min de y: {valores_min['y']:.3f} \t| resultado: {valores_min['valor']:.5f} ")
    elif cont==10000:
        break