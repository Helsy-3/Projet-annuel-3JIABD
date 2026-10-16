"""
test_etape1.py - Teste ETAPE1.basicPerceptron.c depuis Python (ctypes)

Applique le perceptron lineaire sur les 6 cas de tests officiels du prof,
comme test_etape4.py le fait pour le SVM. Le perceptron etant lineaire, il
reussit sur les cas lineaires (Linear Simple, Linear Multiple, Multi Linear
3 classes) et echoue sur les cas non lineaires (XOR, Cross, Multi Cross) -
c'est attendu, ca montre justement la limite du modele lineaire.

Utilisation (depuis la racine du repo), apres avoir compile la bibliotheque :
    python C-algorithms/test_etape1.py
"""

import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
NOM_LIB = "ETAPE1.basicPerceptron.dll" if sys.platform.startswith("win") else "ETAPE1.basicPerceptron.so"
lib = ctypes.CDLL(os.path.join(DOSSIER, NOM_LIB))

P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)

lib.run_perceptron_dataset.argtypes = [P_DOUBLE, P_INT, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_int, P_DOUBLE, P_DOUBLE]
lib.run_perceptron_dataset.restype = ctypes.c_int

lib.perceptron_sauvegarder.argtypes = [ctypes.c_char_p, P_DOUBLE, ctypes.c_double, ctypes.c_int]
lib.perceptron_sauvegarder.restype = ctypes.c_int
lib.perceptron_charger.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, P_INT]
lib.perceptron_charger.restype = ctypes.c_int


class Perceptron:
    """Un perceptron a 2 classes : garde les poids et le biais appris par le C."""

    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.learning_rate, self.max_iter = learning_rate, max_iter

    def entrainer(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_exemples, n_features = X.shape
        self.poids = np.zeros(n_features)
        self.biais = ctypes.c_double(0.0)
        self.erreurs = lib.run_perceptron_dataset(
            X.ctypes.data_as(P_DOUBLE), y.ctypes.data_as(P_INT),
            n_exemples, n_features, self.learning_rate, self.max_iter,
            self.poids.ctypes.data_as(P_DOUBLE), ctypes.byref(self.biais))
        return self

    def score(self, points):
        points = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
        return points @ self.poids + self.biais.value

    def predire(self, points):
        return np.where(self.score(points) >= 0, 1, -1)

    def sauvegarder(self, chemin):
        return lib.perceptron_sauvegarder(chemin.encode(), self.poids.ctypes.data_as(P_DOUBLE),
                                          self.biais.value, len(self.poids))

    @classmethod
    def charger(cls, chemin, n_features):
        perceptron = cls()
        perceptron.poids = np.zeros(n_features)
        biais = ctypes.c_double()
        n = ctypes.c_int()
        if lib.perceptron_charger(chemin.encode(), perceptron.poids.ctypes.data_as(P_DOUBLE),
                                  biais, n) != 0:
            raise IOError(f"Impossible de lire {chemin}")
        perceptron.biais = biais
        return perceptron


class PerceptronMultiClasse:
    """Un contre tous : un perceptron par classe, on garde la classe au score le plus haut."""

    def __init__(self, **parametres):
        self.parametres = parametres

    def entrainer(self, X, y):
        self.classes = np.unique(y)
        self.modeles = [Perceptron(**self.parametres).entrainer(X, np.where(y == c, 1, -1))
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
    print("=== Perceptron lineaire sur les 6 cas de tests officiels ===")
    print("(lineaire = modele reussit ; non lineaire = echec attendu, demontre la limite)\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    cas = [
        ("Linear Simple", (np.array([[1, 1], [2, 3], [3, 3]], dtype=float), np.array([1, -1, -1])),
         Perceptron(), 1.0, True),
        ("Linear Multiple", linear_multiple(), Perceptron(), 0.95, True),
        ("XOR (non lineaire)", (np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float),
         np.array([1, 1, -1, -1])), Perceptron(), 1.0, False),
        ("Cross (non lineaire)", cross(), Perceptron(), 1.0, False),
        ("Multi Linear 3 classes", multi_linear_3_classes(), PerceptronMultiClasse(), 0.90, True),
        ("Multi Cross (non lineaire)", multi_cross(), PerceptronMultiClasse(), 1.0, False),
    ]
    for ax, (nom, (X, y), modele, seuil, doit_reussir) in zip(axes.ravel(), cas):
        modele.entrainer(X, y)
        precision = (modele.predire(X) == y).mean()
        if doit_reussir:
            verifier(f"{nom} : precision {precision:.0%} (attendu : reussite)", precision >= seuil)
        else:
            verifier(f"{nom} : precision {precision:.0%} (echec attendu, non lineaire)", precision < seuil)
        tracer(ax, modele, X, y, f"{nom} - {precision:.0%}")

    fig.tight_layout()
    chemin = os.path.join(DOSSIER, "test_etape1.png")
    fig.savefig(chemin, dpi=110)
    print(f"\nFigure enregistree : {chemin}")

    print("\n=== Sauvegarde / chargement ===")
    modele = Perceptron().entrainer(*linear_multiple())
    chemin_modele = os.path.join(DOSSIER, "perceptron_sauvegarde.txt")
    modele.sauvegarder(chemin_modele)
    recharge = Perceptron.charger(chemin_modele, n_features=len(modele.poids))
    memes_poids = np.allclose(modele.poids, recharge.poids) and np.isclose(modele.biais.value, recharge.biais.value)
    verifier("Modele recharge : memes poids et biais", memes_poids)
    os.remove(chemin_modele)

    print("\n" + ("TOUS LES TESTS PASSENT" if tout_ok else "AU MOINS UN TEST ECHOUE"))
    plt.show()


if __name__ == "__main__":
    main()
