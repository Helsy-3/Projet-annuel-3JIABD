"""
test_etape2.py - Teste ETAPE2.xor.c depuis Python (ctypes)
Auteur : Helsy Soglo

Verifie que :
  1. le perceptron ECHOUE sur XOR brut (cas "KO")
  2. le perceptron REUSSIT sur XOR apres la transformation non lineaire (x1*x2)
  3. la fonction transformer() calcule bien les produits
  4. les fonctions marchent aussi avec des donnees envoyees depuis Python
et trace la frontiere de decision apprise (figure test_etape2.png).

Utilisation (depuis la racine du repo), apres avoir compile la bibliotheque :
    python C-algorithms/test_etape2.py
"""

import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
NOM_LIB = "ETAPE2.xor.dll" if sys.platform.startswith("win") else "ETAPE2.xor.so"
lib = ctypes.CDLL(os.path.join(DOSSIER, NOM_LIB))

# --- On decrit a ctypes les parametres des fonctions C -----------------------
P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)
ARGS_ENTRAINER = [P_DOUBLE, P_INT, ctypes.c_int, ctypes.c_int,
                  ctypes.c_double, ctypes.c_int, P_DOUBLE, P_DOUBLE]

lib.run_xor_brut.restype = ctypes.c_int
lib.run_xor_transforme.restype = ctypes.c_int
lib.nb_features_transformees.argtypes = [ctypes.c_int]
lib.nb_features_transformees.restype = ctypes.c_int
lib.transformer.argtypes = [P_DOUBLE, ctypes.c_int, ctypes.c_int, P_DOUBLE]
lib.transformer.restype = None
lib.entrainer_perceptron.argtypes = ARGS_ENTRAINER
lib.entrainer_perceptron.restype = ctypes.c_int
lib.entrainer_perceptron_transforme.argtypes = ARGS_ENTRAINER
lib.entrainer_perceptron_transforme.restype = ctypes.c_int


def entrainer(X, y, transforme, learning_rate=0.1, max_iter=1000):
    """Appelle le perceptron C. Renvoie (erreurs, poids, biais)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.int32)
    n_exemples, n_features = X.shape
    n_poids = lib.nb_features_transformees(n_features) if transforme else n_features
    poids = np.zeros(n_poids)
    biais = ctypes.c_double(0.0)
    fonction = lib.entrainer_perceptron_transforme if transforme else lib.entrainer_perceptron
    erreurs = fonction(X.ctypes.data_as(P_DOUBLE), y.ctypes.data_as(P_INT),
                       n_exemples, n_features, learning_rate, max_iter,
                       poids.ctypes.data_as(P_DOUBLE), ctypes.byref(biais))
    return erreurs, poids, biais.value


def transformer(X):
    """Appelle la transformation C : ajoute les produits deux a deux."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_exemples, n_features = X.shape
    sortie = np.zeros((n_exemples, lib.nb_features_transformees(n_features)))
    lib.transformer(X.ctypes.data_as(P_DOUBLE), n_exemples, n_features,
                    sortie.ctypes.data_as(P_DOUBLE))
    return sortie


def verifier(nom, condition):
    print(f"[{'OK' if condition else 'ECHEC'}] {nom}")
    return condition


def tracer(ax, X, y, poids, biais, transforme, titre):
    """Colorie le plan selon la prediction du perceptron."""
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grille = np.c_[xx.ravel(), yy.ravel()]
    if transforme:
        grille = transformer(grille)
    z = np.where(grille @ poids + biais >= 0, 1, -1)
    ax.contourf(xx, yy, z.reshape(xx.shape), alpha=0.3, cmap="bwr")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=120)
    ax.set_title(titre)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def main():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float)
    y = np.array([1, 1, -1, -1])
    tout_ok = True

    print("=== Cas de test du prof (donnees dans le fichier C) ===")
    tout_ok &= verifier("XOR brut : le perceptron echoue (erreurs > 0)", lib.run_xor_brut() > 0)
    tout_ok &= verifier("XOR transforme : 0 erreur", lib.run_xor_transforme() == 0)

    print("\n=== Transformation non lineaire ===")
    attendu_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=float)
    tout_ok &= verifier("[x1, x2] -> [x1, x2, x1*x2]", np.array_equal(transformer(X), attendu_2))
    attendu_3 = np.array([[2, 3, 4, 6, 8, 12]], dtype=float)
    tout_ok &= verifier("[2, 3, 4] -> [2, 3, 4, 2*3, 2*4, 3*4]",
                        np.array_equal(transformer([[2, 3, 4]]), attendu_3))

    print("\n=== Donnees envoyees depuis Python ===")
    err_brut, w_brut, b_brut = entrainer(X, y, transforme=False)
    err_tr, w_tr, b_tr = entrainer(X, y, transforme=True)
    tout_ok &= verifier(f"XOR brut : {err_brut}/4 erreurs", err_brut > 0)
    tout_ok &= verifier(f"XOR transforme : {err_tr}/4 erreurs", err_tr == 0)
    print(f"     poids appris : {w_tr}, biais : {b_tr:.2f}")

    # Cas lineairement separable : la transformation ne doit rien casser
    X_lin = np.array([[1, 1], [2, 3], [3, 3]], dtype=float)
    y_lin = np.array([1, -1, -1])
    tout_ok &= verifier("Linear Simple brut : 0 erreur", entrainer(X_lin, y_lin, False)[0] == 0)
    tout_ok &= verifier("Linear Simple transforme : 0 erreur", entrainer(X_lin, y_lin, True)[0] == 0)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    tracer(axes[0], X, y, w_brut, b_brut, False, f"XOR brut : {err_brut}/4 erreurs (KO)")
    tracer(axes[1], X, y, w_tr, b_tr, True, f"XOR + x1*x2 : {err_tr}/4 erreurs (OK)")
    fig.tight_layout()
    chemin = os.path.join(DOSSIER, "test_etape2.png")
    fig.savefig(chemin, dpi=120)
    print(f"\nFigure enregistree : {chemin}")

    print("\n" + ("TOUS LES TESTS PASSENT" if tout_ok else "AU MOINS UN TEST ECHOUE"))
    plt.show()


if __name__ == "__main__":
    main()