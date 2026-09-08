# Projet annuel 

# First step
L'objectif est de tester le modèle sur plusieurs cas simples et connus, avant de les appliquer à la problématique réelle. 
L'objectif est de commencer volontairement avec des problèmes très simples. 

Nous avons créé volontairement deux problèmes simples. 

1. Données linéairement séparables

On teste d'abord le projet avec des données linéairement séparables. 

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [6, 6],
    [6, 7],
    [7, 6],
    [7, 7]
])

y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])

![hey](/img/01.png)

On observe que les deux groupes de données sont séparés. Il est facilement possible de tracer une droite entre les deux groupes. 

2. Données non linéairement séparables

Dans le deuxième cas, les classes qui sont testées sont volontairement différentes. 

Graphiquement :

![hey2](/img/02.png)

On observe que le problème XOR n'est pas linéairement séparable. Le perceptron ne peut pas résoudre XOR car il ne peut pas construire une frontière de décision linéaire. 

Cela insiste sur la nécessité d'utiliser un PMC (Perceptron Multi Couches). Les couches de neurones cachées dans le PMC lui permettent d'apprendre des relations non linéaires. 

3. Perceptron Multi-Couches