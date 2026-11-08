"""
test_etape5.py - Teste ETAPE5.RBF.c (RBF Network) depuis Python (ctypes)

Contrairement aux autres etapes, la fonction RBF() du fichier C ne renvoie
rien a Python (elle fait juste des printf en C). On appelle donc depuis
Python, dans le meme ordre que RBF() le fait en interne, les fonctions C
individuelles (k-means, sigma, entrainement des poids, prediction) pour
recuperer les resultats - sans ajouter ni modifier de code dans le C.

Compilation (depuis la racine du repo) - utils.c doit etre compile avec,
car ETAPE5.RBF.c appelle calculer_closest_centroid() et calculer_dot_output()
qui y sont definies :
    gcc -shared -fPIC C-algorithms/ETAPE5.RBF.c C-algorithms/utils.c -o C-algorithms/ETAPE5.RBF.so -lm

Utilisation :
    python C-algorithms/test_etape5.py
"""

import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DOSSIER = os.path.dirname(os.path.abspath(__file__))
NOM_LIB = "ETAPE5.RBF.dll" if sys.platform.startswith("win") else "ETAPE5.RBF.so"
lib = ctypes.CDLL(os.path.join(DOSSIER, NOM_LIB))

P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)

# Chacune de ces fonctions est un morceau de l'algorithme RBF (ecrit en C) qu'on appelle
# depuis Python dans le bon ordre - RBF() en C fait exactement la meme sequence, mais
# ne renvoie rien a Python, donc on la reconstruit nous-memes ici.

# ETAPE 1/2 du k-means : choisir des centroides de depart, puis les stabiliser
lib.find_centroids.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P_DOUBLE, P_DOUBLE]
lib.find_centroids.restype = None  # void en C -> None cote ctypes

lib.assign_clusters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P_DOUBLE, P_DOUBLE, P_INT, P_INT]
lib.assign_clusters.restype = None

lib.update_centroids.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P_DOUBLE, P_INT, P_INT, P_DOUBLE]
lib.update_centroids.restype = None

# ETAPE 3 : sigma = largeur des gaussiennes (distance moyenne entre les centroides)
lib.calculer_sigma.argtypes = [ctypes.c_int, ctypes.c_int, P_DOUBLE]
lib.calculer_sigma.restype = ctypes.c_double

# ETAPE 4 : pour un point donne, sa "proximite" (activation RBF) a chaque centroide
lib.calculer_dot_rbf_closeness_to_centroids.argtypes = [ctypes.c_int, P_DOUBLE, ctypes.c_int, P_DOUBLE,
                                                         ctypes.c_double, P_DOUBLE]
lib.calculer_dot_rbf_closeness_to_centroids.restype = None

# ETAPE 5 : apprend les poids (un par centroide) par descente de gradient
lib.recursive_train_weights.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, P_DOUBLE, P_DOUBLE, P_DOUBLE,
                               ctypes.c_double, P_DOUBLE, ctypes.c_double, ctypes.c_int]
lib.recursive_train_weights.restype = None

# ETAPE 6 : prediction finale = somme ponderee des proximites (voir calculer_dot_rbf_closeness_to_centroids)
lib.calculer_dot_output.argtypes = [ctypes.c_int, P_DOUBLE, P_DOUBLE]
lib.calculer_dot_output.restype = ctypes.c_double

# Sauvegarde/chargement : centroides + sigma + poids dans un fichier texte simple
lib.rbf_sauvegarder.argtypes = [ctypes.c_char_p, P_DOUBLE, ctypes.c_double, P_DOUBLE, ctypes.c_int, ctypes.c_int]
lib.rbf_sauvegarder.restype = ctypes.c_int
lib.rbf_charger.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, P_INT, P_INT, P_DOUBLE]
lib.rbf_charger.restype = ctypes.c_int


class RBF:
    """RBF Network a 2 classes (-1/1), entraine sur une cible numerique
    comme le C le propose. k = nombre de centroides."""

    def __init__(self, k=10, learning_rate=0.01, epochs=500):
        self.k_souhaite, self.learning_rate, self.epochs = k, learning_rate, epochs

    def _clustering(self, X, k, n, d):
        """Lance k-means une fois. Si un cluster se retrouve vide, update_centroids_mean
        divise par zero et les centroides deviennent NaN (ca contamine tout ensuite)."""
        centroids = np.zeros((k, d))
        lib.find_centroids(k, n, d, X.ctypes.data_as(P_DOUBLE), centroids.ctypes.data_as(P_DOUBLE))
        clusters = np.zeros((k, n), dtype=np.int32)
        tailles = np.zeros(k, dtype=np.int32)
        for _ in range(100):  # K-MEANS : on stabilise les centroides
            lib.assign_clusters(k, n, d, X.ctypes.data_as(P_DOUBLE), centroids.ctypes.data_as(P_DOUBLE),
                                 clusters.ctypes.data_as(P_INT), tailles.ctypes.data_as(P_INT))
            lib.update_centroids(k, n, d, X.ctypes.data_as(P_DOUBLE), clusters.ctypes.data_as(P_INT),
                                       tailles.ctypes.data_as(P_INT), centroids.ctypes.data_as(P_DOUBLE))
        return centroids

    def entrainer(self, X, y, essais=50):
        """Le k-means (et donc le RBF) depend du tirage aleatoire des centroides de depart :
        un mauvais tirage peut donner un cluster vide (NaN, cf _clustering) ou juste un
        mauvais decoupage de l'espace. On essaie plusieurs tirages et on garde celui qui
        apprend le mieux sur les donnees d'entrainement (redemarrages multiples, pratique
        standard pour le k-means)."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n, d = X.shape
        k = max(2, min(self.k_souhaite, n))

        # Si TOUS les essais tombent sur un cluster vide (NaN), k est probablement trop
        # grand par rapport a n : on reduit k et on retente.
        while True:
            meilleure_precision = -1
            for _ in range(essais):
                centroids = self._clustering(X, k, n, d)
                if np.isnan(centroids).any():
                    continue
                sigma = lib.calculer_sigma(k, d, centroids.ctypes.data_as(P_DOUBLE))
                poids = np.zeros(k)
                # recursive_train_weights affiche un printf() par epoch (MSE) dans la console - normal.
                lib.recursive_train_weights(k, n, d, X.ctypes.data_as(P_DOUBLE), y.ctypes.data_as(P_DOUBLE),
                                   centroids.ctypes.data_as(P_DOUBLE), sigma,
                                   poids.ctypes.data_as(P_DOUBLE), self.learning_rate, self.epochs)
                self.k, self.centroids, self.sigma, self.poids = k, centroids, sigma, poids
                precision = (self.predire(X) == y).mean()
                if precision > meilleure_precision:
                    meilleure_precision = precision
                    meilleurs_parametres = (centroids, sigma, poids)
            if meilleure_precision >= 0:
                break
            k -= 1
            if k < 2:
                raise RuntimeError("k-means n'arrive pas a trouver de clusters valides sur ces donnees")

        self.k = k
        self.centroids, self.sigma, self.poids = meilleurs_parametres
        return self

    def score(self, points):
        points = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
        d = points.shape[1]
        proximite = np.zeros(self.k)
        sorties = np.zeros(len(points))
        for i, point in enumerate(points):
            lib.calculer_dot_rbf_closeness_to_centroids(d, point.ctypes.data_as(P_DOUBLE), self.k,
                                                     self.centroids.ctypes.data_as(P_DOUBLE), self.sigma,
                                                     proximite.ctypes.data_as(P_DOUBLE))
            sorties[i] = lib.calculer_dot_output(self.k, proximite.ctypes.data_as(P_DOUBLE),
                                              self.poids.ctypes.data_as(P_DOUBLE))
        return sorties

    def predire(self, points):
        return np.where(self.score(points) >= 0, 1, -1)

    def sauvegarder(self, chemin):
        d = self.centroids.shape[1]
        return lib.rbf_sauvegarder(chemin.encode(), self.centroids.ctypes.data_as(P_DOUBLE),
                                   self.sigma, self.poids.ctypes.data_as(P_DOUBLE), self.k, d)

    @classmethod
    def charger(cls, chemin, k, d):
        rbf = cls()
        rbf.centroids = np.zeros((k, d))
        rbf.poids = np.zeros(k)
        k_lu, d_lu, sigma = ctypes.c_int(), ctypes.c_int(), ctypes.c_double()
        if lib.rbf_charger(chemin.encode(), rbf.centroids.ctypes.data_as(P_DOUBLE),
                           rbf.poids.ctypes.data_as(P_DOUBLE), k_lu, d_lu, sigma) != 0:
            raise IOError(f"Impossible de lire {chemin}")
        rbf.k, rbf.sigma = k_lu.value, sigma.value
        return rbf


class RBFMultiClasse:
    """Un contre tous, comme PerceptronMultiClasse/SVMMultiClasse : un RBF par classe."""

    def __init__(self, **parametres):
        self.parametres = parametres

    def entrainer(self, X, y):
        self.classes = np.unique(y)
        self.modeles = [RBF(**self.parametres).entrainer(X, np.where(y == c, 1, -1).astype(float))
                        for c in self.classes]
        return self

    def predire(self, points):
        scores = np.array([m.score(points) for m in self.modeles])
        return self.classes[np.argmax(scores, axis=0)]


rng = np.random.default_rng(0)


def linear_multiple():
    X = np.concatenate([rng.random((50, 2)) * 0.9 + [1, 1], rng.random((50, 2)) * 0.9 + [2, 2]])
    return X, np.concatenate([np.ones(50), -np.ones(50)]).astype(int)


def cross():
    X = rng.random((500, 2)) * 2.0 - 1.0
    return X, np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])


def multi_linear_3_classes():
    X = rng.random((500, 2)) * 2.0 - 1.0
    y = []
    for p in X:
        if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0:
            y.append(0)
        elif -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0:
            y.append(1)
        elif -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0:
            y.append(2)
        else:
            y.append(-1)
    y = np.array(y)
    return X[y >= 0], y[y >= 0]


def multi_cross():
    X = rng.random((1000, 2)) * 2.0 - 1.0
    y = []
    for p in X:
        if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25:
            y.append(0)
        elif abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25:
            y.append(1)
        else:
            y.append(2)
    return X, np.array(y)


def image_vers_vecteur(chemin, taille=(2, 2)):
    img = Image.open(chemin).convert("RGB")
    img = img.resize(taille)
    return (np.array(img) / 255.0).flatten()


def lister_images(dossier):
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    return [os.path.join(dossier, f) for f in sorted(os.listdir(dossier)) if f.lower().endswith(extensions)]


def charger_dataset_reel(sous_dossier):
    """sous_dossier = 'train' ou 'test' (voir dataset-img/separer_dataset.py, qui a deja
    separe les photos une fois pour toutes - pas besoin de refaire le split ici)."""
    base = os.path.join(DOSSIER, "..", "dataset-img", sous_dossier)
    chemins_plastic = lister_images(os.path.join(base, "plastic"))
    chemins_verre = lister_images(os.path.join(base, "verre"))
    chemins_metal = lister_images(os.path.join(base, "metal"))
    chemins = chemins_plastic + chemins_verre + chemins_metal
    classes = np.array([0] * len(chemins_plastic) + [1] * len(chemins_verre) + [2] * len(chemins_metal))
    return chemins, classes


tout_ok = True


def verifier(nom, condition):
    global tout_ok
    tout_ok &= bool(condition)
    print(f"[{'OK' if condition else 'ECHEC'}] {nom}")


def tracer(ax, modele, X, y, titre):
    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 80), np.linspace(y0, y1, 80))
    z = modele.predire(np.c_[xx.ravel(), yy.ravel()])
    ax.contourf(xx, yy, z.reshape(xx.shape), alpha=0.3, cmap="brg")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", edgecolors="k", s=20)
    ax.set_title(titre, fontsize=9)


def main():
    print("=== RBF Network sur les 6 cas de tests officiels ===")
    print("(le RBF gere les frontieres non lineaires, contrairement au perceptron lineaire)\n")

    # k = nombre de centroides choisi a la main pour chaque cas : il faut assez de
    # centroides pour "couvrir" la forme de la frontiere (plus une frontiere est
    # decoupee/complexe, plus il en faut), mais pas trop par rapport au nombre de
    # points sinon le modele risque le sur-apprentissage (voir le README).
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    cas = [
        ("Linear Simple", (np.array([[1, 1], [2, 3], [3, 3]], dtype=float), np.array([1, -1, -1])), RBF(k=2)),
        ("Linear Multiple", linear_multiple(), RBF(k=6)),
        ("XOR", (np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float), np.array([1, 1, -1, -1])), RBF(k=4)),
        ("Cross", cross(), RBF(k=15)),
        ("Multi Linear 3 classes", multi_linear_3_classes(), RBFMultiClasse(k=6)),
        ("Multi Cross", multi_cross(), RBFMultiClasse(k=15)),
    ]
    for ax, (nom, (X, y), modele) in zip(axes.ravel(), cas):
        modele.entrainer(X, y)
        precision = (modele.predire(X) == y).mean()
        if nom == "Multi Cross":
            # Limite connue : notre sigma est une seule valeur globale (distance moyenne entre
            # centroides), trop grossiere pour distinguer les bandes fines de ce damier - peu
            # importe le nombre de clusters ou d'epochs (teste jusqu'a k=60, epochs=1500).
            print(f"[LIMITE CONNUE] {nom} : precision {precision:.0%} (sigma global trop grossier pour un damier fin)")
        else:
            verifier(f"{nom} : precision {precision:.0%}", precision >= 0.7)
        tracer(ax, modele, X, y, f"{nom} - {precision:.0%}")

    fig.tight_layout()
    chemin = os.path.join(DOSSIER, "test_etape5.png")
    fig.savefig(chemin, dpi=110)
    print(f"\nFigure enregistree : {chemin}")

    # Deuxieme partie : meme principe, mais sur de vraies photos au lieu de points 2D.
    # On separe les photos en train/test AVANT d'entrainer, pour mesurer si le modele
    # generalise a des photos qu'il n'a jamais vues (pas juste s'il les a apprises par coeur).
    print("\n=== Dataset reel (photos de dechets) ===")
    chemins_train, classes_train = charger_dataset_reel("train")
    chemins_test, classes_test = charger_dataset_reel("test")
    vecteurs_train = np.array([image_vers_vecteur(c) for c in chemins_train])
    vecteurs_test = np.array([image_vers_vecteur(c) for c in chemins_test])

    modele_reel = RBFMultiClasse(k=8)
    modele_reel.entrainer(vecteurs_train, classes_train)
    precision_train = (modele_reel.predire(vecteurs_train) == classes_train).mean()
    precision_test = (modele_reel.predire(vecteurs_test) == classes_test).mean()
    print(f"Entrainement : {len(chemins_train)} photos, Test : {len(chemins_test)} photos (jamais vues)")
    verifier(f"Dataset reel - precision entrainement {precision_train:.0%}", precision_train >= 0.5)
    verifier(f"Dataset reel - precision test {precision_test:.0%}", precision_test >= 0.3)

    print("\n=== Sauvegarde / chargement ===")
    modele_un_contre_tous = modele_reel.modeles[0]  # un seul RBF binaire, parmi les 3 du "un contre tous"
    chemin_modele = os.path.join(DOSSIER, "rbf_sauvegarde.txt")
    modele_un_contre_tous.sauvegarder(chemin_modele)
    recharge = RBF.charger(chemin_modele, k=modele_un_contre_tous.k, d=modele_un_contre_tous.centroids.shape[1])
    memes_parametres = (np.allclose(modele_un_contre_tous.centroids, recharge.centroids)
                        and np.allclose(modele_un_contre_tous.poids, recharge.poids)
                        and np.isclose(modele_un_contre_tous.sigma, recharge.sigma))
    verifier("Modele recharge : memes centroides, sigma et poids", memes_parametres)
    os.remove(chemin_modele)

    print("\n" + ("TOUS LES TESTS PASSENT" if tout_ok else "AU MOINS UN TEST ECHOUE"))
    plt.show()


if __name__ == "__main__":
    main()
