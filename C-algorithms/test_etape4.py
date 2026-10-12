import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
NOM_LIB = "ETAPE4.svm.dll" if sys.platform.startswith("win") else "ETAPE4.svm.so"
lib = ctypes.CDLL(os.path.join(DOSSIER, NOM_LIB))

LINEAIRE, RBF = 0, 1

P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)

lib.svm_entrainer.argtypes = [P_DOUBLE, P_INT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int, P_DOUBLE]
lib.svm_entrainer.restype = ctypes.c_int
lib.svm_score.argtypes = [P_DOUBLE, P_DOUBLE, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_double, P_DOUBLE]
lib.svm_score.restype = ctypes.c_double
lib.svm_sauvegarder.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.svm_sauvegarder.restype = ctypes.c_int
lib.svm_lire_entete.argtypes = [ctypes.c_char_p, P_INT, P_INT, P_INT, P_DOUBLE]
lib.svm_lire_entete.restype = ctypes.c_int
lib.svm_charger.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE]
lib.svm_charger.restype = ctypes.c_int
for nom in ("run_svm_lineaire", "run_svm_xor_lineaire", "run_svm_xor_rbf"):
    getattr(lib, nom).restype = ctypes.c_int


def ptr(tableau):
    return tableau.ctypes.data_as(P_DOUBLE)


class SVM:
    """Un SVM a 2 classes : garde les donnees et les coefficients appris par le C."""

    def __init__(self, noyau=LINEAIRE, gamma=1.0, C=1000.0, max_iter=10000):
        self.noyau, self.gamma, self.C, self.max_iter = noyau, gamma, C, max_iter

    def entrainer(self, X, y):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        self.coefs = np.zeros(len(self.X))
        self.n_supports = lib.svm_entrainer(
            ptr(self.X), y.ctypes.data_as(P_INT), len(self.X), self.X.shape[1],
            self.noyau, self.gamma, self.C, self.max_iter, ptr(self.coefs))
        return self

    def score(self, points):
        points = np.ascontiguousarray(np.atleast_2d(points), dtype=np.float64)
        return np.array([lib.svm_score(ptr(self.X), ptr(self.coefs), len(self.X),
                                       self.X.shape[1], self.noyau, self.gamma, ptr(p))
                         for p in points])

    def predire(self, points):
        return np.where(self.score(points) >= 0, 1, -1)

    def sauvegarder(self, chemin):
        return lib.svm_sauvegarder(chemin.encode(), ptr(self.X), ptr(self.coefs), len(self.X),
                                   self.X.shape[1], self.noyau, self.gamma)

    @classmethod
    def charger(cls, chemin):
        n, d, noyau = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        gamma = ctypes.c_double()
        if lib.svm_lire_entete(chemin.encode(), n, d, noyau, gamma) != 0:
            raise IOError(f"Impossible de lire {chemin}")
        svm = cls(noyau.value, gamma.value)
        svm.X = np.zeros((n.value, d.value))
        svm.coefs = np.zeros(n.value)
        svm.n_supports = lib.svm_charger(chemin.encode(), ptr(svm.X), ptr(svm.coefs))
        return svm


class SVMMultiClasse:
    """Un contre tous : un SVM par classe, on garde la classe au score le plus haut."""

    def __init__(self, **parametres):
        self.parametres = parametres

    def entrainer(self, X, y):
        self.classes = np.unique(y)
        self.svms = [SVM(**self.parametres).entrainer(X, np.where(y == c, 1, -1))
                     for c in self.classes]
        return self

    def predire(self, points):
        scores = np.array([svm.score(points) for svm in self.svms])
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
    if isinstance(modele, SVM):
        sv = modele.coefs != 0
        ax.scatter(X[sv, 0], X[sv, 1], s=110, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.set_title(titre, fontsize=9)


def main():
    print("=== Cas de test dans le fichier C ===")
    verifier("Linear Simple, noyau lineaire : 0 erreur", lib.run_svm_lineaire() == 0)
    verifier("XOR, noyau lineaire : echoue (erreurs > 0)", lib.run_svm_xor_lineaire() > 0)
    verifier("XOR, noyau RBF : 0 erreur", lib.run_svm_xor_rbf() == 0)

    print("\n=== Donnees envoyees depuis Python ===")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    cas = [
        ("Linear Simple", (np.array([[1, 1], [2, 3], [3, 3]], dtype=float), np.array([1, -1, -1])),
         SVM(LINEAIRE), 1.0),
        ("Linear Multiple", linear_multiple(), SVM(LINEAIRE), 1.0),
        ("XOR", (np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float), np.array([1, 1, -1, -1])),
         SVM(RBF, gamma=1.0), 1.0),
        ("Cross", cross(), SVM(RBF, gamma=5.0), 0.98),
        ("Multi Linear 3 classes", multi_linear_3_classes(), SVMMultiClasse(noyau=LINEAIRE), 0.98),
        ("Multi Cross", multi_cross(), SVMMultiClasse(noyau=RBF, gamma=30.0), 0.98),
    ]
    for ax, (nom, (X, y), modele, seuil) in zip(axes.ravel(), cas):
        modele.entrainer(X, y)
        precision = (modele.predire(X) == y).mean()
        detail = f", {modele.n_supports} vecteurs de support" if isinstance(modele, SVM) else ""
        verifier(f"{nom} : precision {precision:.0%}{detail}", precision >= seuil)
        tracer(ax, modele, X, y, f"{nom} - {precision:.0%}{detail}")

    print("\n=== Sauvegarde / chargement ===")
    X, y = cross()
    svm = SVM(RBF, gamma=5.0).entrainer(X, y)
    chemin = os.path.join(DOSSIER, "test_svm_modele.txt")
    ecrits = svm.sauvegarder(chemin)
    recharge = SVM.charger(chemin)
    os.remove(chemin)
    verifier(f"{ecrits} vecteurs de support sauvegardes sur {len(X)} points",
             ecrits == svm.n_supports and ecrits < len(X))
    points = rng.random((200, 2)) * 2.0 - 1.0
    verifier("Modele recharge : memes scores",
             np.allclose(svm.score(points), recharge.score(points)))

    fig.tight_layout()
    chemin_fig = os.path.join(DOSSIER, "test_etape4.png")
    fig.savefig(chemin_fig, dpi=110)
    print(f"\nFigure enregistree : {chemin_fig}")
    print("\n" + ("TOUS LES TESTS PASSENT" if tout_ok else "AU MOINS UN TEST ECHOUE"))
    plt.show()


if __name__ == "__main__":
    main()