# Projet annuel 

Contexte - Vocabulaire 

* CLASSE = Quand nous parlons de classe, nous désignons une catégorie de déchet que le modèle doit reconnaître.

Exemple: 
- Classe 0 : plastique 🧴
- Classe 1 : verre 🍾
- Classe 2 : métal 🥫


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

L’idée est surtout de montrer que le MLP peut apprendre une séparation non linéaire entre les classes de déchets.
Le modèle linéaire permet de rechercher une frontière de décision linéaire entre les différentes catégories. 
Cependant, les caractéristiques visuelles des déchets ne sont pas nécessairement séparables linéairement. 
Le perceptron multicouche introduit une ou plusieurs couches cachées ainsi que des fonctions d'activation non linéaires, 
ce qui permet au modèle d'apprendre des relations plus complexes entre les caractéristiques et les classes.

Première étape : Rassembler des photos de déchets 
Pour entraîner le PMC, il faut un dataset d'images étiquetées : chaque photo doit appartenir à une classe.

Deuxième étape: transformer les photos en vecteurs 
Un PMC ne peut pas directement comprendre un fichier .jpg.
Il lui faut des nombres.

Pour commencer simplement, on peut transformer chaque image en une petite image, par exemple 32 × 32 pixels, puis la transformer en un vecteur.

Troisième étape: 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Tu prends tes images et tu les sépares :
80 % → entraînement
20 % → test


Création du PMC
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

* hidden_layer_sizes=(32, 16) signifie donc deux couches cachées, une de 32 neurones et une de 16.

* activation="relu" veut dire que les neurones des couches cachées utilisent la fonction ReLU (Rectified Linear Unit).
Elle est très simple :

ReLU(x)=max(0,x)

Donc : 
si x < 0  →  0
si x ≥ 0  →  x

SI 
x = -3  → ReLU(x) = 0
x = -0.5 → ReLU(x) = 0
x = 2   → ReLU(x) = 2
x = 5   → ReLU(x) = 5

Cette fonction est très importante pour le projet, car elle permet au PMC d'apprendre des relations non linéaires.
Sans fonction d'activation non linéaire, empiler plusieurs couches reviendrait essentiellement à faire encore une transformation linéaire.
En gros, ReLU introduit la non-linéarité nécessaire au perceptron multicouche pour apprendre des relations complexes entre les caractéristiques des images.


mlp.fit(X_train, y_train)

C'est ici que le PMC apprend à partir de tes photos.

photo → classe correcte

y_pred = mlp.predict(X_test)


# Difficultés rencontrées
Pour le choix des images, attention à ne pas prendre que des images avec un fond blanc, de peur que le modèle fasse de fausses prédictions. 
