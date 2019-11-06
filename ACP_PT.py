# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:45:41 2019

@author: Alejandro Molina
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#Carga de datos
#Abrir el archivo .csv 
data = r'C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\ACP.csv'
df = pd.read_csv(data, names = ['Aceites y grasas','Alimentos de origen animal',
                                'Azucares','Bebidas','Cereales','Comida mexicana',
                                'Frutas','Leches','Legumbres','Verduras','ID'])

n = len(df.columns)                      #Contar columnas
m = len(list(csv.reader(open(data))))    #Contar filas

# Se divide la matriz del dataset en dos partes
X = df.iloc[:,0:n-1]
# la submatriz x contiene los valores de las primeras 10 columnas del dataframe y todas las filas
y = df.iloc[:,n-1].values
# El vector y contiene los valores de la 11a columna (paciente) para todas las filas

#Normalizacion de los valores
df_ = X - X.mean()

#Calculo de la matriz de covarianza
mat_cov = np.cov(df_.T)

#Calculo vectores propios y valores propios
eigenvalores, eigenvectores = np.linalg.eig(mat_cov)

#Hacemos una lista por parejas (eigenvalor, eigenvector) 
eig_par = [(np.abs(eigenvalores[i]), eigenvectores[:,i]) for i in range(len(eigenvalores))]

# Orden de las tuplas (eigenvalor, eigenvector) de mayor a menor
eig_par.sort(key=lambda x: x[0], reverse=True)

## Lista de eigenvalores en orden descendiente y formacion del FeatureVector
#for i in eig_par:
#    print(i[0])
#    print(i[1])

#A partir de los valores propios, calculamos la varianza explicada
tot = sum(eigenvalores)
var_exp = [(i / tot)*100 for i in sorted(eigenvalores, reverse=True)]
sum_var_exp = np.cumsum(var_exp)

#Representamos en un diagrama de barras la varianza por cada eigenvalor, y la acumulada
with plt.style.context('seaborn-pastel'):
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Valor propio')
    ax1.set_xlabel('Componentes Principales')
    ax1.bar(range(n-1), eigenvalores/10, alpha=0.5, align='center',
            label='Contribucion del componente', color='g')
    ax1.tick_params(axis='y')
    #Instancia un segundo eje que comparte al eje x
    ax2 = ax1.twinx()
    ax2.plot(range(n-1), sum_var_exp, 'o-', label='Varianza acumulada (%)')
    ax2.set_ylabel('Porcentaje de Varianza Explicada')
    ax2.set_xlabel('Componentes Principales')
    ax2.tick_params(axis='y')
    plt.savefig('CP.jpg')
    plt.legend(loc='best')
    plt.tight_layout()

    
#Creamos el FeatureVector = [eig1, eig2, ...]
fv = np.hstack((eig_par[0][1].reshape(n-1, 1), eig_par[1][1].reshape(n-1, 1)))

#Calculo de los datos finales, FinalData = RowFeatureVector x RowDataAdjust
rfv = np.transpose(fv)
rda = np.transpose(df_)
fd = np.dot(rfv, rda)
fd = np.transpose(fd)
fd = pd.DataFrame(fd.tolist())

# Grafica de datos finales
fig2, axx = plt.subplots()
axx.axhline(0, color='black')
axx.axvline(0, color='black')
axx.scatter(fd[0], fd[1], color='red')
#ID de cada paciente
for i, txt in enumerate(y):
    axx.annotate(txt, (fd[0][i], fd[1][i]))
plt.grid()
plt.ylabel('CP2')
plt.xlabel('CP1')
fig2.savefig('PCA.jpg')

 