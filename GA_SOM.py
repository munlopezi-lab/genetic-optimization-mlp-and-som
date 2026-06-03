# Import necessary libraries

#from random import seed
#from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score
import pygad
import warnings
from sklearn.exceptions import ConvergenceWarning
from minisom import MiniSom
from sklearn.cluster import KMeans 
import numpy as np


warnings.filterwarnings("ignore", category=ConvergenceWarning)  #ignora los warnings de convergencia

seed = 55


gene_space = [
    {'low': 5, 'high': 30},          #límites para el ancho del mapa
    {'low': 5, 'high': 30},          #límites para el alto del mapa
    {'low': 0.1, 'high': 5.0},       #valores para el radio de vecindad
    {'low': 0.01, 'high': 0.1},      #rango de la tasa de aprendizaje
    [0, 1, 2, 3],                    #función de vecindad (0: 'gaussian', 1: 'mexican_hat', 2: 'bubble', 3: 'triangle')
    {'low': 500, 'high': 10000}      #límites para el número de iteraciones
]


wine = load_wine()                                   #cargamos el dataset de vino
scale = MinMaxScaler(feature_range=(0, 1))           #creamos un objeto de la clase MinMaxScaler para escalar los datos, esto es importante para que el modelo funcione bien, especialmente si estás usando una función de activación como la sigmoide o la tangente hiperbólica, ya que estas funciones de activación pueden saturarse si los valores de entrada son muy grandes o muy pequeños, lo que puede hacer que el modelo no aprenda correctamente, al escalar los datos entre 0 y 1, evitamos este problema y ayudamos al modelo a aprender mejor.
X = scale.fit_transform(wine.data)                   #escalas los datos para que estén entre 0 y 1, esto es importante para que el modelo funcione bien, especialmente si estás usando una función de activación como la sigmoide o la tangente hiperbólica
X, y = X, wine.target                                #separas los datos en características (X) y etiquetas (y), esto es importante para que el modelo pueda aprender a predecir las etiquetas a partir de las características, sin esta línea, el modelo no podría aprender a predecir las etiquetas a partir de las características, lo que hace que el modelo sea inútil, ya que no podría usar los datos para nada
n_features = X.shape[1]                              #obtenemos el número de características, esto es importante para definir la arquitectura del modelo, especialmente el número de neuronas en la capa de entrada, sin esta línea, no podrías definir la arquitectura del modelo correctamente, lo que puede hacer que el modelo no funcione bien o incluso que no funcione en absoluto




 #divides los datos en conjuntos de entrenamiento y prueba, esto es importante para evaluar el rendimiento del modelo, sin esta línea, no podrías evaluar el rendimiento del modelo, lo que hace que el modelo sea inútil, ya que no podrías usar los resultados obtenidos por el modelo para nada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)# es tu semilla para reproducibilidad



def decode_chromosome(x):   #esta función decodifica el individuo, es decir, convierte los valores del individuo a los parámetros del modelo, el primer valor del individuo es el número de capas ocultas, los siguientes 10 valores son el número de neuronas en cada capa oculta, el siguiente valor es la función de activación, el siguiente valor es el solver, el siguiente valor es la tasa de aprendizaje, el siguiente valor es el número máximo de iteraciones y el último valor es el momentum

    map_width = int(round(x[0]))          #esto es para determinar el ancho del mapa, el número de capas ocultas se determina por el primer valor del individuo
    map_height = int(round(x[1]))         #esto es para determinar el alto del mapa, el número de capas ocultas se determina por el primer valor del individuo
    learning_rate = x[3]                  #esto es para determinar la tasa de aprendizaje
#esto es para determinar la función de vecindad, el número de capas ocultas se determina por el primer valor del individuo
    match int(round(x[4])):                                               #esto es para determinar la función de activación, el número de capas ocultas se determina por el primer valor del individuo
        case 0:
            neighborhood_funcs = 'gaussian'
        case 1:
            neighborhood_funcs = 'mexican_hat'
        case 2:            
            neighborhood_funcs = 'bubble'
        case 3:            
            neighborhood_funcs = 'triangle'
        case _:
            neighborhood_funcs = 'gaussian'  #valor por defecto en caso de que el valor redondeado sea distinto a los esperados 
    

    if neighborhood_funcs in ['bubble', 'triangle']:
        sigma = max(1, int(round(x[2])))
    else:
        sigma = x[2]
    max_iter = int(round(x[5]))              #esto es para determinar el número máximo de iteraciones


    return map_width, map_height, sigma, learning_rate, neighborhood_funcs, max_iter


def fitness_fnc(ga_instance, individuos, solution_idx):                                                 #esta es la función para evaluar la aptitud de cada individuo, es decir, la precisión del modelo con los parámetros definidos por el individuo, el primer valor del individuo es el número de capas ocultas, los siguientes 10 valores son el número de neuronas en cada capa oculta, el siguiente valor es la función de activación, el siguiente valor es el solver, el siguiente valor es la tasa de aprendizaje, el siguiente valor es el número máximo de iteraciones y el último valor es el momentum
    map_width, map_height, sigma, learning_rate, neighborhood_funcs, max_iter = decode_chromosome(individuos)  #decodificas el individuo para obtener los parámetros del modelo


    som = MiniSom(x=map_width,                          #ancho del mapa
            y=map_height,                               #alto del mapa
            input_len=n_features,                       #número de características de entrada
            sigma=sigma,                                #radio de vecindad
            learning_rate=learning_rate,                #tasa de aprendizaje
            neighborhood_function=neighborhood_funcs,   #función de vecindad
            random_seed=seed)                           # semilla para reproducibilidad
    
    som.pca_weights_init(X)                                          # Inicialización de pesos 
    som.train_batch(X, num_iteration=max_iter, verbose=False)        # Entrenamiento del SOM                                      #haces las predicciones con los datos de prueba
    kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
    weights = som.get_weights().reshape(-1, n_features)
    kmeans.fit(weights)

    labels_kmeans = kmeans.labels_
    ganadores = np.array([som.winner(d) for d in X])

    cluster_labels = np.array([labels_kmeans[g[0] * map_height + g[1]] for g in ganadores])
    score = adjusted_rand_score(y, cluster_labels)
    return max(0.0, score)  #devuelves la precisión del modelo, asegurándote de que no sea negativa, esto es importante porque la aptitud debe ser un valor positivo, si el valor de score es negativo, se devuelve 0.0 para evitar problemas con la optimización del algoritmo genético, sin esta línea, podrías obtener valores negativos de aptitud.


def main():
   
    ga = pygad.GA(num_generations=50,           #número de generaciones para ejecutar el algoritmo genético
                  num_parents_mating=2,         #número de padres que se seleccionan para reproducirse en cada generación, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  fitness_func=fitness_fnc,     #función de aptitud que se usa para evaluar la calidad de cada individuo en la población, en este caso, es la precisión del modelo con los parámetros definidos por el individuo
                  sol_per_pop=30,               #número de soluciones (individuos) en la población, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  num_genes=6,                 #número de genes en cada individuo, en este caso, es el número de parámetros que se están optimizando, el primer gen es el número de capas ocultas, los siguientes 10 genes son el número de neuronas en cada capa oculta, el siguiente gen es la función de activación, el siguiente gen es el solver, el siguiente gen es la tasa de aprendizaje, el siguiente gen es el número máximo de iteraciones y el último gen es el momentum
                  gene_space=gene_space,        #espacio de genes para cada individuo, en este caso, es una lista de diccionarios que definen el rango de valores posibles para cada gen
                  crossover_probability=0.9,    #probabilidad de cruce entre los padres seleccionados para reproducirse, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  mutation_probability=0.1,     #probabilidad de mutación de los genes en los individuos generados por el cruce, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  random_seed=seed)             #semilla para reproducibilidad, esto es importante para que los resultados sean consistentes entre ejecuciones, si no se establece una semilla, los resultados pueden variar cada vez que se ejecute el algoritmo genético, lo que dificulta la comparación de resultados y la depuración del código
    
   
    ga.run()                                                    #ejecuta el algoritmo genético, esto es lo que realmente hace que el algoritmo genético funcione, sin esta línea, el algoritmo genético no haría nada, simplemente se inicializaría y luego se detendría sin hacer nada
    
    solution, solution_fitness, _ = ga.best_solution()          #obtiene la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías obtener la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    
    print("Mejor solución encontrada: ", (solution))            #imprime la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    print("Fitness de la mejor solución: ", solution_fitness)   #imprime la aptitud de la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la aptitud de la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver la aptitud de la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada

    solucion_decodificada = decode_chromosome(solution)             #decodifica la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, decodificada a los parámetros del modelo, sin esta línea, no podrías usar la mejor solución encontrada por el algoritmo genético para nada, ya que no podrías decodificarla a los parámetros del modelo, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    print("Parámetros del modelo correspondientes a la mejor solución: ", solucion_decodificada )  #imprime los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    #print("Entrenamiento finalizado.")
    #ga.plot_fitness() # Esto generará una gráfica de la evolución

main()