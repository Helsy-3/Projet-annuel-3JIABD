"""
test_etape3.py - Teste ETAPE3.multiLayersPerceptron.c depuis Python (ctypes)

Verifie que :
  1. le PMC (Perceptron Multi-Couches) reussit sur les 3 exemples de test fixes
  2. le PMC reussit aussi sur une portion du vrai dataset (photos de dechets),
     avec un vrai split entrainement/test (run_pmc_dataset() renvoie maintenant
     les poids appris, qu'on reutilise avec predire_pmc() sur les photos de test
     jamais vues pendant l'entrainement)

Utilisation (depuis la racine du repo), apres avoir compile la bibliotheque :
    python C-algorithms/test_etape3.py
"""

import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DOSSIER = os.path.dirname(os.path.abspath(__file__))
NOM_LIB = "ETAPE3.multiLayersPerceptron.dll" if sys.platform.startswith("win") else "ETAPE3.multiLayersPerceptron.so"
lib = ctypes.CDLL(os.path.join(DOSSIER, NOM_LIB))

P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)

# run_pmc_dataset ENTRAINE le reseau (W1/b1/W2/b2 sont des parametres de SORTIE : le C
# les remplit avec les poids appris, qu'on peut ensuite reutiliser nous-memes en Python
# avec predire_pmc() ci-dessous, sans jamais ré-entrainer).
lib.run_pmc_dataset.argtypes = [P_DOUBLE, P_INT, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, P_DOUBLE, P_DOUBLE, P_DOUBLE, P_DOUBLE]
lib.run_pmc_dataset.restype = ctypes.c_int

# predire_pmc ne fait QUE predire (forward pass), a partir de poids deja entraines -
# c'est ce qui permet d'evaluer sur des photos de test jamais vues pendant l'entrainement.
lib.predire_pmc.argtypes = [P_DOUBLE, P_DOUBLE, P_DOUBLE, P_DOUBLE, P_DOUBLE,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.predire_pmc.restype = ctypes.c_int

lib.pmc_sauvegarder.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, P_DOUBLE, P_DOUBLE,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.pmc_sauvegarder.restype = ctypes.c_int
lib.pmc_charger.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, P_DOUBLE, P_DOUBLE,
                            P_INT, P_INT, P_INT]
lib.pmc_charger.restype = ctypes.c_int

H_DIM, OUT_DIM = 8, 3


def sauvegarder_pmc(chemin, poids, in_dim=12):
    W1, b1, W2, b2 = poids
    h_dim = len(b1)  # retrouve depuis la taille reelle des poids, pas depuis la constante H_DIM
    return lib.pmc_sauvegarder(chemin.encode(), W1.ctypes.data_as(P_DOUBLE), b1.ctypes.data_as(P_DOUBLE),
                               W2.ctypes.data_as(P_DOUBLE), b2.ctypes.data_as(P_DOUBLE),
                               in_dim, h_dim, OUT_DIM)


def charger_pmc(chemin, in_dim=12):
    W1, b1 = np.zeros(in_dim * H_DIM), np.zeros(H_DIM)
    W2, b2 = np.zeros(H_DIM * OUT_DIM), np.zeros(OUT_DIM)
    n_in, n_h, n_out = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if lib.pmc_charger(chemin.encode(), W1.ctypes.data_as(P_DOUBLE), b1.ctypes.data_as(P_DOUBLE),
                       W2.ctypes.data_as(P_DOUBLE), b2.ctypes.data_as(P_DOUBLE), n_in, n_h, n_out) != 0:
        raise IOError(f"Impossible de lire {chemin}")
    return (W1, b1, W2, b2)


def entrainer_pmc(vecteurs, classes, in_dim=12, h_dim=H_DIM, epochs=10000):
    """Entraine le PMC en C. h_dim = taille de la couche cachee (la "complexite" du
    modele - plus elle est grande, plus le PMC peut apprendre de details, quitte a
    apprendre le bruit par coeur si elle est trop grande vu le peu de photos qu'on a).
    epochs = nombre de passages sur les donnees d'entrainement.
    Renvoie (bonnes_predictions_entrainement, poids)."""
    vecteurs = np.ascontiguousarray(vecteurs, dtype=np.float64)
    classes = np.ascontiguousarray(classes, dtype=np.int32)
    n_exemples = len(vecteurs)
    W1, b1 = np.zeros(in_dim * h_dim), np.zeros(h_dim)
    W2, b2 = np.zeros(h_dim * OUT_DIM), np.zeros(OUT_DIM)
    bonnes = lib.run_pmc_dataset(
        vecteurs.ctypes.data_as(P_DOUBLE), classes.ctypes.data_as(P_INT),
        n_exemples, in_dim, h_dim, OUT_DIM, epochs,
        W1.ctypes.data_as(P_DOUBLE), b1.ctypes.data_as(P_DOUBLE),
        W2.ctypes.data_as(P_DOUBLE), b2.ctypes.data_as(P_DOUBLE))
    return bonnes, (W1, b1, W2, b2)


def predire_pmc(vecteur, poids, in_dim=12):
    """Predit la classe d'un seul exemple a partir de poids DEJA entraines (pas d'entrainement ici).
    h_dim n'est pas un parametre : on le retrouve directement depuis la taille de b1."""
    W1, b1, W2, b2 = poids
    h_dim = len(b1)
    vecteur = np.ascontiguousarray(vecteur, dtype=np.float64)
    return lib.predire_pmc(
        vecteur.ctypes.data_as(P_DOUBLE),
        W1.ctypes.data_as(P_DOUBLE), b1.ctypes.data_as(P_DOUBLE),
        W2.ctypes.data_as(P_DOUBLE), b2.ctypes.data_as(P_DOUBLE),
        in_dim, h_dim, OUT_DIM)


def image_vers_vecteur(chemin, taille=(2, 2)):
    img = Image.open(chemin).convert("RGB")
    img = img.resize(taille)
    return (np.array(img) / 255.0).flatten()


def lister_images(dossier):
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    return [os.path.join(dossier, f) for f in sorted(os.listdir(dossier)) if f.lower().endswith(extensions)]


def separer_train_test(chemins, classes, proportion_test=0.2, graine=0):
    """Split stratifie : au moins 1 exemple de test par classe, meme si la classe est petite."""
    rng_split = np.random.default_rng(graine)
    chemins = np.array(chemins)
    train_idx, test_idx = [], []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        rng_split.shuffle(idx)
        n_test = max(1, round(len(idx) * proportion_test))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return (chemins[train_idx].tolist(), classes[train_idx],
            chemins[test_idx].tolist(), classes[test_idx])


tout_ok = True


def verifier(nom, condition):
    global tout_ok
    tout_ok &= bool(condition)
    print(f"[{'OK' if condition else 'ECHEC'}] {nom}")


def main():
    print("=== Cas de test du prof (donnees dans le fichier C) ===")
    vecteurs_test = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ], dtype=np.float64)
    vraies_classes_test = np.array([2, 0, 1], dtype=np.int32)
    bonnes_test, _ = entrainer_pmc(vecteurs_test, vraies_classes_test)
    verifier(f"Cas de test : {bonnes_test}/3 bonnes predictions", bonnes_test == 3)

    print("\n=== Donnees reelles envoyees depuis Python (photos de dechets) ===")
    chemins_plastic = lister_images(os.path.join(DOSSIER, "..", "dataset-img", "plastic"))
    chemins_verre = lister_images(os.path.join(DOSSIER, "..", "dataset-img", "verre"))
    chemins_metal = lister_images(os.path.join(DOSSIER, "..", "dataset-img", "metal"))
    chemins = chemins_plastic + chemins_verre + chemins_metal
    classes = np.array([0] * len(chemins_plastic) + [1] * len(chemins_verre) + [2] * len(chemins_metal))

    chemins_train, classes_train, chemins_test, classes_test = separer_train_test(chemins, classes)
    vecteurs_train = np.array([image_vers_vecteur(c) for c in chemins_train])
    vecteurs_test = np.array([image_vers_vecteur(c) for c in chemins_test])

    bonnes_train, poids = entrainer_pmc(vecteurs_train, classes_train)
    precision_train = bonnes_train / len(chemins_train)

    predictions_test = np.array([predire_pmc(v, poids) for v in vecteurs_test])
    precision_test = (predictions_test == classes_test).mean()

    print(f"Entrainement : {len(chemins_train)} photos, Test : {len(chemins_test)} photos (jamais vues)")
    verifier(f"Dataset reel - precision entrainement {precision_train:.0%}", precision_train >= 0.5)
    verifier(f"Dataset reel - precision test {precision_test:.0%}", precision_test >= 0.3)

    plt.figure(figsize=(4, 3))
    plt.bar(["Cas de test", "Entrainement", "Test"],
            [bonnes_test / 3, precision_train, precision_test], color=["steelblue", "steelblue", "green"])
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.title("PMC : cas de test, entrainement et test")
    chemin_fig = os.path.join(DOSSIER, "test_etape3.png")
    plt.savefig(chemin_fig, dpi=110)
    print(f"\nFigure enregistree : {chemin_fig}")

    print("\n=== Sauvegarde / chargement ===")
    chemin_modele = os.path.join(DOSSIER, "pmc_sauvegarde.txt")
    sauvegarder_pmc(chemin_modele, poids)
    poids_recharges = charger_pmc(chemin_modele)
    memes_poids = all(np.allclose(a, b) for a, b in zip(poids, poids_recharges))
    verifier("Modele recharge : memes poids", memes_poids)
    os.remove(chemin_modele)

    print("\n" + ("TOUS LES TESTS PASSENT" if tout_ok else "AU MOINS UN TEST ECHOUE"))
    plt.show()


if __name__ == "__main__":
    main()
