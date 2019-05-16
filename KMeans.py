
# @author: Ana Aguilera, Jordi Gimenez, Agustin Molina

"""
     =============================================================================
     ==========================  A T E N C I O N =================================
     La funcion Plot ha sido modificada con tal de proporcionar una representacion
     mas intuitiva de como funciona el algoritmo sin embargo su funcionamiento es
     similar a su version original. Si la opcion verbose esta en false estas
     visualizaciones no se mostraran.
     Ademas otra funcion ha sido creada para ver el resultado final de este.
     La funcion iterate se ha modificado debido a un error del codigo original donde
     el valor de la opcion max_iter no era tenido en cuenta. Ademas se ha cambiado
     los lugares donde se llama a la funcion plot para mostrar cada paso del algoritmo.
     =============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA


def NIUs():
    return 1496128, 1425098, 1493035


def distance(X, C):
    """
        Esta función es la encargada de generar una matriz bidimensional donde cada eje representa
        una lista de puntos n-dimensionales distinta. Dicha matriz almacena en cada celda de esta
        la distancia euclidiana entre cada punto de las dos listas.

        Argumentos:
            - X: Primera lista de puntos n-dimensionales.
            - C: Segunda lista de puntos n-dimensionales.

        Retorna:
            - Matriz con la longitud de cada lista en sendos ejes.

    """

    # Generamos una matriz vacia con el tamaño necesario para almacenar nuestros datos.
    distances = np.empty(shape=(X.shape[0], C.shape[0]))

    # Para cada punto en la primera lista...
    for point_idx in range(distances.shape[0]):
        # ...calculamos su distancia euclideana contra cada uno de los centroides
            distances[point_idx][:] = np.sqrt(np.sum((X[point_idx][:] - C[:][:])**2, axis=1))
    return distances


class KMeans():

    def __init__(self, X, K, options=None):
        """
            Separa el constructor de la clase KMeans en tres metodos separados.

            Argumentos:
                - X: Matriz n-dimensional que almacena los datos de la imagen a tratar.
                - K: Numero (Entero) de centroides.
                - options: Diccionario que almacena las caracteristicas de como se ejecutara el programa.

        """

        self._init_X(X)
        self.original_image = X
        self._init_options(options)
        self._init_rest(K)

    def _init_X(self, X):
        """
            Este metodo es el encargado de transformar la imagen en un formato normalizado
            con el que poder trabajar de manera uniforme. Esto se refiere que sin importar
            como venga la imagen entendemos que la ultima dimension hace referencia al caracter
            de color del elemento indivisible de la imagen.

            Con esto lo que trataremos es transformar esta matriz en una lista donde cada punto
            almacene la informacion cromatica de cada punto independientemente de las dimensiones
            de color que esta requiera.

            |1	R 	G 	B 	|
            |2	R 	G 	B 	|
            |.	R 	G 	B 	|
            |.	R 	G 	B 	|
            |.	R 	G 	B 	|
            |N	R 	G 	B 	|

            Argumentos:
                - X: Matriz con la informacion de la imagen (Suele venir en formato Anchura*Altura*RGB).

            Retorna:
                - Esta funcion no retorna valor alguno. El resultado de ejecutar este metodo se almacena
                en la variable propia de la clase X.

        """


        # En la unica linea de este codigo utiliza la funcion de la libreria numpy reshape.
        # A esta funcion le decimos que la matriz X cambiara su forma a una matriz bidimensional
        # donde el segundo eje tendra el tamaño del numero de canales que utilicemos (X.shape[-1])
        # para informar sobre un color. Mientras tanto la primera dimension sera el colapso de todas
        # las otras dimensiones que no sean la de los canales de color (indicado con un -1).
        self.X = X.reshape([-1, X.shape[-1]])

    def _init_options(self, options):
        """
            Esta funcion es la responsable de inicializar nuestro atributo propio options.
            En este diccionario de opciones lo que haremos sera almacenar como quiere el usuario
            que se ejecute nuestro programa. Dichas opciones son las siguientes:

            - km_init -> Como deseamos que se inicialicen los centroides.

            - verbose -> Indica si queremos graficar los resultados de nuestro algoritmo.

            - tolerance -> Esta variable establece que tan cerca han de estar los centroides de su
                iteracion anterior para considerar que ya no se estan desplazando.

            - max_iter -> Cuantas iteraciones maximas del algoritmo KMeans puede realizar nuestro programa.

            - fitting -> Que metodo para evaluar la eficacia del valor K utilizamos.

            Argumentos:
                - options: Diccionario cuya clave es el concepto de la opcion
                    y el valor representa como lo establecemos.

            Retorno:
                - Esta funcion no retorna nada, sin embargo los cambios de opciones
                    son almacenados en la variable propia options.

        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 10
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options

    def _init_rest(self, K):
        """
            Esta funcion se encarga de inicializar el resto de estructuras de datos necesarias para ejecutar
            el algoritmo KMeans, todo esto en funcion del valor de K.

            Si K <= 0 directamente no se inicializara ninguna estructura pues mas tarde la funcion best_k
            se encargara de realizar varias inicializaciones distintas.

            Argumentos:
             - K: Numero de clusters que queremos en esta ejecucion completa del algoritmo. Si este valor
                es igual o inferior a 0 esta funcion no realizara ningun cambio.

        """

        self.K = K
        if self.K > 0:
            self._init_centroids()
            self.old_centroids = np.empty_like(self.centroids) # Old centroids se usa para calcular el cambio entre iteraciones.
            self.clusters = np.zeros(len(self.X)) # Lista donde almacenaremos a que centroide pertenece cada punto.
            self._cluster_points()
        self.num_iter = 0  # Esta variable mantiene la cuenta de cuantas iteracioes se llevaremos.

    def _init_centroids(self):
        """
            Esta metodo inicializara los valores de nuestros centroides.
            La manera en la que los inicialicemos viene definida por la opcion km_init.

            Si esta opcion es random establecera el valor de los canales RGB de manera aleatoria.
            En caso de ser first, elegira los K primeros puntos no repetidos.

        """

        if self.options['km_init'].lower() == 'random':
            # Aqui creamos la matriz de centroides de K*nCanales con un valor aleatorio entre 0 y 255.
            np.random.seed()
            self.centroids = np.random.rand(self.K, self.X.shape[1])*255
        else:
            if self.options['km_init'].lower() == 'first':
                # Creamos la matriz centroides con tamaño K*nCanales y sin valores en sus celdas.
                self.centroids = np.empty([self.K, self.X.shape[-1]])

                # Points sera nuestro iterador de la lista de puntos de la imagen.
                point = 0
                ctr = 0
                while ctr < self.K:
                    # ...si nuestro punto no esta en centroides...
                    if self.X[point] not in self.centroids:
                        # ...se lo asignaremos.
                        self.centroids[ctr] = self.X[point]
                        ctr += 1

                    # Pasamos al siguiente punto
                    point += 1


    def _cluster_points(self):
        """
            Este metodo asigna a cada punto su centroide mas cercano.
            Esto lo almacenamos en la variable clusters la cual es una
            matriz monodimensional donde cada valor puede ir desde 0 hasta k.

        """

        ## Teniendo en cuenta que distance nos devuelve una matriz de nPuntos*nCentroides
        # lo que haremos es calcular el valor minimo de cada fila (horizontalmente, axis=0) lo cual
        # hara colapsar la matriz distancia a una sola dimension de orden nPuntos. Sin embargo
        # en lugar de almacenar el valor minimo de dicha fila lo que almacenaremos sera el numero
        # de su columna el cual coincide con el indice del centroide.

        self.clusters = distance(self.X, self.centroids).argmin(axis=1)

    def _get_centroids(self):
        """
            Aqui nos encargaremos de realizar el recalculo de la posicion de los centroides.

            Una vez ya hemos asignado un cluster a cada punto procedemos a desplazar el centroide
            de su posicion original al centro de masas de cada punto que en terminos sencillos
            seria simplemente la posicion promedio de cada punto.

        """

        # Primero actualizamos nuestra variable old_centroids, pues seguidamente centroids sera modificada.
        self.old_centroids = np.copy(self.centroids)

        # Por cada centroide...
        for ctr in range(self.K):
            # ...si este es cluster de algun punto existente...
            if self.X[self.clusters == ctr].shape[0] > 0:
                # ...calcula el promedio de todos los puntos que pertenezcan a dicho cluster.

                ## Esto se realiza igualando clusters a al indice del centroide. Esto nos
                # devolvera una lista de Trues y Falses. Utilizando esta lista booleana
                # como indice de la lista de puntos X, esta ultima devolvera una lista
                # con los valores para los indices donde en la primera lista habia True.
                # Seguidamente calcula dentro de esta ultima lista el promedio pero de manera
                # vertical (axis=0). Con lo que de una matriz nPuntos*nCanales colapsara
                # en una lista de orden nCanales la cual sera almacenada en el centroide iterado.
                self.centroids[ctr] = np.mean(self.X[self.clusters == ctr], axis=0)
            else:
                self.centroids[ctr] = np.random.rand(1, self.X.shape[1])*255


    def _converges(self):
        """
            Esta funcion se encarga de comprobar si nuestro algoritmo ha terminado. Eso significa que los centroides
            ya no se desplazan de manera significativa respecto a su posicion en la anterior iteracion.

            Por lo tanto si al menos un centroide se sigue desplazando como minimo en una distancia dada
            por la opcion tolerance, consideraremos que nuestro algoritmo no ha convergido.

            Retorna:
                - converges: booleano que nos indica si hemos convergido o no.
        """

        return np.amax(np.linalg.norm(self.centroids - self.old_centroids, axis=1)) < self.options['tolerance']

    def _iterate(self):
        """
            Funcion que reune todas las funcionalidades que debe realizar nuestro algoritmo en una iteracion.

        """

        # Incrementamos el contador de iteracion.
        self.num_iter += 1
        # Asignamos cada punto a un cluster.
        self._cluster_points()
        # Recalculamos la posicion de los centroides.
        self._get_centroids()

    def run(self):
        """
            Una vez construido el objeto kmeans, esta sera el metodo al que llamaremos
            desde nuestro codigo main para que nuestro algoritmo comience a ejecutarse.

        """

        # En caso de que K sea 0 pues realizaremos el calculo de cual es el numero optimo de K.
        if self.K == 0:
            self.bestK()
            return

        # Si verbose es true ploteamos los datos
        if self.options['verbose']:
            self.show_image()
            self.plot()

        # Iteramos una primera vez sin fijarnos en la convergencia
        self._iterate()

        # Si el numero de iteraciones no supere al maximo y mientras no hayamos convergido.
        while self.options['max_iter'] > self.num_iter and self._converges() == False:

            # Si verbose es true ploteamos los datos
            if self.options['verbose'] and self.num_iter != 0:
                self.plot()

            # Realizaremos una nueva iteracion del algoritmo.
            self._iterate()

        self.show_image()


    def bestK(self):
        """
            Este metodo se encarga de calcular cual es el valor optimo de K para tener
            una buena agrupacion y encontrar los colores(grupos) mas relevantes.
        """

        fisher_results = [] # Variable donde guardaremos los resultados de optimalidad de dicha K
        best_k = 0 # Contador de que K es la actual
        cmp = True # Almacenador de la comparacion heuristica por la cual determinamos que ya no hay mejor K

        # Mientras best_k menor o igual a 3 o la heuristica nos diga que pueden haber mejores K...
        while best_k < 3 or cmp:
            # ...pasa a la siguiente K...
            best_k += 1

            # ...inicializa el programa con la nueva K...
            self._init_rest(best_k)
            # ...ejecuta el programa...
            self.run()
            # ...y almacena su optimalidad.
            fisher_results.append(self.fitting())

            # Si cumplimos un minimo de iteraciones...
            if best_k >= 3:
                # ...comprueba que el 3 veces el cambio entre un punto y su anterior es mas grande que el de este punto y el siguiente.
                cmp = 3*(fisher_results[best_k-1] - fisher_results[best_k-2]) < (fisher_results[best_k-2] - fisher_results[best_k-3])
        best_k = best_k-1
        return best_k

    def fitting(self):
        """
            Este metodo se encarga de evaluar que tan bien se han agrupado nuestros datos.
            Esto lo realizamos con el discriminante de Fisher.

            Este valor es en resumidas cuentas la razon entre la distancia media entre cada
            punto y su centrode respecto a la distancia media entre cada centroide respecto
            al centro de todos los puntos en conjunto.

        """

        # Si nuestra opcion de valorar K es fisher...
        if self.options['fitting'].lower() == 'fisher':
            # ...calcula las distancias de entre los puntos y sus centroides...
            dist_p2c = distance(self.X, self.centroids)
            # ...y la distancia entre los centroides y el centro de todos los puntos.
            dist_c2c = distance(self.centroids, np.mean(self.X, axis=0))

            avg_intra_dist = 0 # Numerador
            avg_inter_dist = 0 # Denominador

            # Por cada centroide...
            for ctr in range(self.K):
                # ...si este es cluster de algun punto existente...
                if self.X[self.clusters == ctr].shape[0] > 0:
                    # ...calcula la distancia entre todos los puntos y el centroide.
                    avg_intra_dist += np.sum(dist_p2c[:, ctr][self.clusters == ctr])*(1/self.X[self.clusters == ctr].shape[0])*(1/self.K)
                # Finalmente calcula la media de cada centroide al centro.
                avg_inter_dist += np.sum(dist_c2c[ctr])*(1/self.K)

            # Devuelve la razon entre la intra distancia y la interdistancia.
            return avg_intra_dist / avg_inter_dist

        else:
            # Si la opcion de fitting no es fisher sencillamente devuelve un valor aleatorio entre 0 y 1.
            return np.random.rand(1)

    def plot(self):
        """
            Esta funcion es la encargada de plottear todos los datos de nuestra funcion.
            Esta funcion ha sido modificada con la intencion de comunicar de una manera mas
            intuitiva que esta ocurriendo en cada iteracion del bucle.

        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=self.X / 255, marker='.', alpha=0.3)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c=self.centroids / 255, marker='o', s=5000, alpha=0.75, linewidths=1, edgecolors="k")


        textdict = "K: "+str(self.K)+"\nInit: "+str(self.options["km_init"]+"\nIter: "+str(self.num_iter))
        box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text( 200, 100, 200, textdict, transform=ax.transAxes, fontsize=16, horizontalalignment="left", verticalalignment="bottom", bbox=box)

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')

        plt.show()

    def show_image(self):
        """
            Esta funcion muestra la imagen que se esta procesando. Ademas aplica un filtro de imagen
            haciendo que cada punto se transforme en el color de su centroide.
        """

        if self.num_iter == 0:
            plt.imshow(self.original_image)
        else:
            new_image = np.copy(self.X)
            for ctr in range(self.K):
                if new_image[self.clusters == ctr].shape[0] > 0:
                    new_image[self.clusters == ctr] = self.centroids[ctr]
            plt.imshow(new_image.reshape(self.original_image.shape))
            plt.show()
            plt.pause(1)