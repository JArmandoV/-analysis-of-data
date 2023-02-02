file = 'C:/Users/Arman/OneDrive/Escritorio/Datos/prueba.csv'
#Copie y pegue la ruta de su carpita con el divisor / y escriba correctamente el nombre del archivo
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(file)

columna = df["Gasto nacional bruto (% del PIB)"]
columna2 = df["Población urbana"]


#print(df.head()) #head and tail tienen por default n = 5
#print(df.tail())
#print("Numero de variables y numero de registros")
#print(df.shape) #tupla con numero de renglones y numero de columnas
#pd.set_option("display.max_rows",None, "display.max_columns",None)
#print("Lista de las columnas")
#print(df.columns) #Lista con nombres de las columnas
#print("Tipos de datos")
#print(df.dtypes) #tipos de datos de cada columna

#print("Valores unicos: ")
#print(df["Crecimiento de la población (% anual)"].unique())

#quita los renglones (axis=0) que contiene cualquier (how='any','all') columna vacia, inplace significa
#df.dropna(axis = 0, how = 'any', inplace = True)




#print("Valores Maximos del crecimiento de poblacion anual ")
#print(columna.max()) #valor maximo
#print("Valores Minimos del crecimiento de poblacion anual")
#print(columna.min()) #valor minimo
print("Desviacion Estandar del crecimiento de Gasto nacional bruto")
print(columna.std()) #desviacion estandar, que tan dispersos estan los datos al rededor de la media
print("Media del crecimiento de Gasto nacional bruto: ")
print(columna.mean()) #promedio
print("Mediana: del crecimiento de Gasto nacional bruto")
print(columna.median()) #mediana (el valor que se encuentra al centro de la lista ordenada

#print("Valores Maximos de la poblacion urbana ")
#print(columna2.max()) #valor maximo
#print("Valores Minimos de la poblacion urbana: ")
print(columna2.min()) #valor minimo
print("Desviacion Estandar de la poblacion urbana: ")
print(columna2.std()) #desviacion estandar, que tan dispersos estan los datos al rededor de la media
print("Media de la poblacion urbana: ")
print(columna2.mean()) #promedio
print("Mediana de la poblacion urbana: ")
print(columna2.median()) #mediana (el valor que se encuentra al centro de la lista ordenada



print("Cajas y bigotes: ")

df.boxplot(column=["Gasto nacional bruto (% del PIB)","PIB per cápita (US$ a precios actuales)"],showcaps=True, grid=True)
plt.show()

df.boxplot(column=["Población urbana"],showcaps=True, grid=True)

plt.show()


 #estadisticas basicos en formato de tabla

#print(df.describe())

import matplotlib.pyplot as plt

df.hist(column="Ingreso nacional bruto (ING) (US$)", color="darkturquoise")
plt.show()
df.hist(column="Gasto nacional bruto (US$ a precios constantes de 2010)", color="darkturquoise")
plt.show()
pd.set_option("display.max_rows",None, "display.max_columns",None)
print(df.corr(numeric_only=True))      # es un dataframe


##############################################################
# Mapa de calor
##############################################################

import seaborn as sns

# vmin int, valor mínimo de la escala
# vmax int, valor mínimo de la escala
# annot boolean, muestra el valor de la correlación, default = False
# cmap mapa de color, colores usados para el mapa, https://matplotlib.org/stable/gallery/color/colormap_reference.html

plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only=True), vmin=-1, vmax=1, annot=True,cmap="RdYlGn");
plt.show()


# Triángulo
import numpy as np
plt.figure(figsize=(15, 5))
mask = np.tril(np.ones_like(df.corr(numeric_only=True), dtype=bool))     # tril para triángulo superior
sns.heatmap(df.corr(numeric_only=True), mask=mask, vmin=-1, vmax=1, annot=True, cmap='Greens')
plt.show()


#Observar una sola variable


##############################################################
#  k means
##############################################################
from sklearn.cluster import KMeans #instalar scikit-learn

test = df[["Gasto nacional bruto (% del PIB)","Población urbana"]]
test = test.dropna(axis = 0, how = 'any')

kmeans = KMeans(n_clusters=3,n_init='auto').fit(test)
centroids = kmeans.cluster_centers_
# print(centroids)
# 
# # Predicciones (cuál es la clase) de acuerdo a los centros calculados
# 
cla = kmeans.predict(test)                   # obtiene las clases de los datos iniciales
# print(cla)
# 
# # Predicción para un nuevo dato
# data = {'danceability': ['.27'], 'energy': [.5]}
# newdf = pd.DataFrame(data)  
# print(kmeans.predict(newdf))
# 
# 
plt.scatter(df["Gasto nacional bruto (% del PIB)"],df["Población urbana"],c=cla)
for i in range(len(centroids)):
    plt.scatter(centroids[i][0],centroids[i][1],marker="*",c="red")
plt.show()