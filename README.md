# Algoritmos-Geneticos-con-pygad-
Los códigos se centran en la optimización de hiperparámetros.

La optimización de hace con la ayuda la biblioteca pygad, donde se definen algunos parámetros como número de generaciones e individuos por generación. Cada individuo presenta una configuración aleatoria de los hiperparámetros definidos en el espacio de el espacio de genes (gene_space) donde se definen rangos de valores que puede tomar cada hiperparámetro.

Cuando genera nuevos individuos (parámetro: num_parents_mating=2) estos resultan ser convinaciones de los dos "padres" seleccionados aleatoriamente. Dicha combinación puede asignar valores inválidos para algunos hiperprámetros. Para solventar esto, definimos la función decode_chromosome, esta nos permite conseguir que cada individuo contenga valores validos de hiperparámetros antes de pasarlos al constructor de la clase MLPClassifier o MiniSom.

Para medir la aptitud de cada individuo se creó la función fitness_fnc, quien mide la aptitud del individuo según que tan alto (bajo) sea el accuracy (o adjusted_rand_score).

IMPORTANTE
en los parámeros que recibe pygad.GA() es necesario indicar el número exacto de genes que hay en nuestro espacio genético, en este caso num_genes=6. 
Además, es necesario indicar el espacio de genes que estamos utilizando en este caso gene_space.
gene_space=gene_space.
