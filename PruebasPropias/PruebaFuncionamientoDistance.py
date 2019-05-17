import numpy as np
np.set_printoptions(precision = 2)

pixels = np.random.rand(15,3)*255
centroids = np.random.rand(3,3)*255

print("Esta es la matriz de centorides 3x3 (K=3 y RGB):")
print(centroids)
print("\n\n")

print("Esta es la matriz RGB de una imagen de 15 pixeles en total:")
print(pixels)
print("\n\n")


print("Deberemos obtener una matriz 15x3 donde tengamos la distancia de cada pixel a cada centroide:")
distances=np.empty(shape=(pixels.shape[0], centroids.shape[0]))
for p in range(distances.shape[0]):
    distances[p][:]=np.sqrt(np.sum((pixels[p][:]-centroids[:][:])**2, axis=1))
print(distances)
print("\n\n")

print("Lista que inidca a que centroide pertenece (más cercano) cada pixel:")
relationPixCen = np.empty(shape=(pixels.shape[0],1))
relationPixCen = np.argmin(distances, axis=1)
print(relationPixCen)
print("\n\n\n")

print("Lista de pixeles pertenecientes a cada centroide (formacion de cluster): ")
oldCentroids = np.copy(centroids)
for cluster in range(0,3):
    print("---> Cluster " + str(cluster) +":")
    clusterPoints=pixels[relationPixCen==cluster]
    print(clusterPoints)    

    print("Medias: " + str(np.mean(clusterPoints, axis=0)))
    print("\n")
    centroids[cluster] = np.mean(clusterPoints, axis=0)
print("Los nuevos centroides son: ")
print(centroids)
print("\n")
print("Respecto a los antiguos centroides, que eran: ")
print(oldCentroids)
print("\n\n")

for ctr in range(0,3):
    distDes = np.sqrt(np.sum((centroids[ctr]-oldCentroids[ctr])**2))
    print("Distancia desplazada por el centroide " + str(ctr)+ ": " + str(distDes))
    print("False") if distDes < 200 else print("True")