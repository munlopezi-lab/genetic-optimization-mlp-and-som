#from random import seed
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score
import pygad
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  #ignora los warnings de convergencia

seed = 55


gene_space = [
    {'low': 1, 'high': 10},        #num_layers
    {'low': 1, 'high': 64},        #capa 1
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},
    {'low': 1, 'high': 64},        #capa 10
    [0,1,2,3],                     #activation
    [0,1,2],                       #solver
    {'low': 0.0001, 'high': 0.01}, #learning_rate
    {'low': 50, 'high': 1000},     #max_iter
    {'low': 0.0, 'high': 1.0}      #momentum
]


wine = load_wine()                                   #cargamos el dataset de vino
scale = MinMaxScaler(feature_range=(0, 1))           #creamos un objeto de la clase MinMaxScaler para escalar los datos, esto es importante para que el modelo funcione bien, especialmente si estás usando una función de activación como la sigmoide o la tangente hiperbólica, ya que estas funciones de activación pueden saturarse si los valores de entrada son muy grandes o muy pequeños, lo que puede hacer que el modelo no aprenda correctamente, al escalar los datos entre 0 y 1, evitamos este problema y ayudamos al modelo a aprender mejor.
X = scale.fit_transform(wine.data)                   #escalas los datos para que estén entre 0 y 1, esto es importante para que el modelo funcione bien, especialmente si estás usando una función de activación como la sigmoide o la tangente hiperbólica
X, y = X, wine.target                                #separas los datos en características (X) y etiquetas (y), esto es importante para que el modelo pueda aprender a predecir las etiquetas a partir de las características, sin esta línea, el modelo no podría aprender a predecir las etiquetas a partir de las características, lo que hace que el modelo sea inútil, ya que no podría usar los datos para nada



 #divides los datos en conjuntos de entrenamiento y prueba, esto es importante para evaluar el rendimiento del modelo, sin esta línea, no podrías evaluar el rendimiento del modelo, lo que hace que el modelo sea inútil, ya que no podrías usar los resultados obtenidos por el modelo para nada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)# es tu semilla para reproducibilidad



def decode_chromosome(x):   #esta función decodifica el individuo, es decir, convierte los valores del individuo a los parámetros del modelo, el primer valor del individuo es el número de capas ocultas, los siguientes 10 valores son el número de neuronas en cada capa oculta, el siguiente valor es la función de activación, el siguiente valor es el solver, el siguiente valor es la tasa de aprendizaje, el siguiente valor es el número máximo de iteraciones y el último valor es el momentum

    num_layers = int(round(x[0]))                                          #número de capas ocultas
    hidden_layers_sizes = tuple([max(int(round(v)), 1) for v in x[1:10]])  #esto es un array con el número de neuronas en cada capa oculta, el número de capas ocultas se determina por el primer valor del individuo
    match int(round(x[11])):                                               #esto es para determinar la función de activación, el número de capas ocultas se determina por el primer valor del individuo
        case 0:
            activation = 'identity'
        case 1:
            activation = 'logistic'
        case 2:            
            activation = 'tanh'
        case 3:            
            activation = 'relu'
        case _:
            activation = 'relu'  #valor por defecto en caso de que el valor redonde a un número fuera del rango de 0 a 3, esto es importante para evitar errores en caso de que el valor redondeado sea un número fuera del rango de 0 a 3, lo que podría causar un error al intentar usar una función de activación que no existe, al establecer un valor por defecto, evitamos este problema y aseguramos que el modelo siempre tenga una función de activación válida, incluso si el valor del individuo no es perfecto.
    

    match int(round(x[12])):     #esto es para determinar el solver, el número de capas ocultas se determina por el primer valor del individuo
        case 0:
            solver = 'lbfgs'
        case 1:
            solver = 'sgd'
        case 2:            
            solver = 'adam'
        case _:            
            solver = 'adam'  #valor por defecto en caso de que el valor redonde a un número fuera del rango de 0 a 2, esto es importante para evitar errores en caso de que el valor redondeado sea un número fuera del rango de 0 a 2, lo que podría causar un error al intentar usar un solver que no existe, al establecer un valor por defecto, evitamos este problema y aseguramos que el modelo siempre tenga un solver válido, incluso si el valor del individuo no es perfecto.
    

    learning_rate = x[13]                        #esto es para determinar la tasa de aprendizaje, el número de capas ocultas se determina por el primer valor del individuo
    max_iter = int(round(x[14]))                 #esto es para determinar el número máximo de iteraciones, el número de capas ocultas se determina por el primer valor del individuo
    momentum = min(1.0, x[15])                   #esto es para determinar el momentum, el número de capas ocultas se determina por el primer valor del individuo


    return hidden_layers_sizes, activation, solver, learning_rate, max_iter, momentum  #devuelves los parámetros del modelo


def fitness_fnc(ga_instance, individuos, solution_idx):                                                 #esta es la función para evaluar la aptitud de cada individuo, es decir, la precisión del modelo con los parámetros definidos por el individuo, el primer valor del individuo es el número de capas ocultas, los siguientes 10 valores son el número de neuronas en cada capa oculta, el siguiente valor es la función de activación, el siguiente valor es el solver, el siguiente valor es la tasa de aprendizaje, el siguiente valor es el número máximo de iteraciones y el último valor es el momentum
    hidden_layers_sizes, activation, solver, learning_rate, max_iter, momentum = decode_chromosome(individuos)  #decodificas el individuo para obtener los parámetros del modelo

    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers_sizes,       #tamaño de las capas ocultas
                            activation=activation,                      #función de activación
                            solver=solver,                              #solver para optimización
                            learning_rate_init=learning_rate,           #tasa de aprendizaje inicial
                            max_iter=max_iter,                          #número máximo de iteraciones 
                            momentum=momentum,                          #momentum para el solver sgd, no afecta a los otros solvers, pero no causa error si se usa con ellos, por lo que lo dejamos para que el algoritmo genético pueda optimizarlo también, aunque no tenga efecto en algunos casos
                            early_stopping=True,random_state=seed)                        #habilita el early stopping para evitar el sobreajuste, esto es importante para que el modelo no se sobreajuste a los datos de entrenamiento, especialmente si el número máximo de iteraciones es alto, al habilitar el early stopping, el modelo se detendrá automáticamente si la precisión en los datos de validación no mejora durante un número determinado de iteraciones, lo que ayuda a evitar el sobreajuste y mejora la generalización del modelo   
    
    mlp.fit(X_train, y_train)                                           #entrenas el modelo con los datos de entrenamiento estandarizados
    y_pred = mlp.predict(X_test)                                        #haces las predicciones con los datos de prueba
    return accuracy_score(y_test, y_pred)                               #devuelves la precisión de las predicciones



def main():
   
    ga = pygad.GA(num_generations=50,           #número de generaciones para ejecutar el algoritmo genético
                  num_parents_mating=2,         #número de padres que se seleccionan para reproducirse en cada generación, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  fitness_func=fitness_fnc,     #función de aptitud que se usa para evaluar la calidad de cada individuo en la población, en este caso, es la precisión del modelo con los parámetros definidos por el individuo
                  sol_per_pop=30,               #número de soluciones (individuos) en la población, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  num_genes=16,                 #número de genes en cada individuo, en este caso, es el número de parámetros que se están optimizando, el primer gen es el número de capas ocultas, los siguientes 10 genes son el número de neuronas en cada capa oculta, el siguiente gen es la función de activación, el siguiente gen es el solver, el siguiente gen es la tasa de aprendizaje, el siguiente gen es el número máximo de iteraciones y el último gen es el momentum
                  gene_space=gene_space,        #espacio de genes para cada individuo, en este caso, es una lista de diccionarios que definen el rango de valores posibles para cada gen
                  crossover_probability=0.9,    #probabilidad de cruce entre los padres seleccionados para reproducirse, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  mutation_probability=0.1,     #probabilidad de mutación de los genes en los individuos generados por el cruce, esto es importante para mantener la diversidad genética en la población, si es muy bajo, el algoritmo puede converger prematuramente a una solución subóptima, si es muy alto, el algoritmo puede tardar más en converger, pero puede encontrar una mejor solución
                  random_seed=seed)             #semilla para reproducibilidad, esto es importante para que los resultados sean consistentes entre ejecuciones, si no se establece una semilla, los resultados pueden variar cada vez que se ejecute el algoritmo genético, lo que dificulta la comparación de resultados y la depuración del código
    
   
    ga.run()                                                    #ejecuta el algoritmo genético, esto es lo que realmente hace que el algoritmo genético funcione, sin esta línea, el algoritmo genético no haría nada, simplemente se inicializaría y luego se detendría sin hacer nada
    
    solution, solution_fitness, _ = ga.best_solution()          #obtiene la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías obtener la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    
    print("Mejor solución encontrada: ", (solution))            #imprime la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    print("Fitness de la mejor solución: ", solution_fitness)   #imprime la aptitud de la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la aptitud de la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver la aptitud de la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada

    solucion_decodificada = decode_chromosome(solution)         #decodifica la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, la mejor solución encontrada por el algoritmo genético, decodificada a los parámetros del modelo, sin esta línea, no podrías usar la mejor solución encontrada por el algoritmo genético para nada, ya que no podrías decodificarla a los parámetros del modelo, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
    print("Parámetros del modelo correspondientes a la mejor solución: ", solucion_decodificada )  #imprime los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, esto es lo que realmente te interesa, los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, sin esta línea, no podrías ver los parámetros del modelo correspondientes a la mejor solución encontrada por el algoritmo genético, lo que hace que el algoritmo genético sea inútil, ya que no podrías usar los resultados obtenidos por el algoritmo genético para nada
  
    #ga.plot_fitness() #gráfica de la evolución

main()