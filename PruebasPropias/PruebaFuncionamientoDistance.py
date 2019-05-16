import numpy as np
np.set_printoptions(precision = 2)

pixels = np.random.randint(0,255,(10,3))
centroids = np.random.randint(0,255,(3,3))

print("Esta es la matriz de centorides 3x3 (K=3 y RGB):")
print(centroids)
print("\n\n\n")

print("Esta es la matriz RGB de una imagen de 10 pixeles en total:")
print(pixels)
print("\n\n\n")


print("Deberemos obtener una matriz 10x3 donde tengamos la distancia de cada pixel a cada centroide")
distances=np.empty(shape=(pixels.shape[0], centroids.shape[0]))
for p in range(distances.shape[0]):
    #print(pixels[p][:]-centroids[:][:])
    distances[p][:]=np.sqrt(np.sum((pixels[p][:]-centroids[:][:])**2, axis=1))
print(distances)