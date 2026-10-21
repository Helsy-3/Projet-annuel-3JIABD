"""
entrainer_modele.py - Entraine le SVM (ETAPE4.svm.c) sur le dataset reel et sauvegarde
un modele par classe (un contre tous) dans app/modeles/, pour que server.py puisse les
charger et predire sur de nouvelles photos, sans avoir a tout ré-entrainer a chaque fois.

Utilisation (depuis la racine du repo) :
    python app/entrainer_modele.py
"""

import ctypes
import os

import numpy as np
from PIL import Image

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)

NOM_LIB = "ETAPE4.svm.so"
lib = ctypes.CDLL(os.path.join(RACINE, "C-algorithms", NOM_LIB))

P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT = ctypes.POINTER(ctypes.c_int)

LINEAIRE, RBF = 0, 1

lib.svm_entrainer.argtypes = [P_DOUBLE, P_INT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int, P_DOUBLE]
lib.svm_entrainer.restype = ctypes.c_int
lib.svm_sauvegarder.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.svm_sauvegarder.restype = ctypes.c_int

NOMS_CLASSES = {0: "plastique", 1: "verre", 2: "metal"}


def image_vers_vecteur(chemin, taille=(2, 2)):
    img = Image.open(chemin).convert("RGB")
    img = img.resize(taille)
    return (np.array(img) / 255.0).flatten()


def lister_images(dossier):
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    return [os.path.join(dossier, f) for f in sorted(os.listdir(dossier)) if f.lower().endswith(extensions)]


def main():
    # On entraine seulement sur dataset-img/train/ (pas test/), comme partout ailleurs
    # dans le projet - voir dataset-img/separer_dataset.py.
    dataset_dir = os.path.join(RACINE, "dataset-img", "train")
    chemins_plastic = lister_images(os.path.join(dataset_dir, "plastic"))
    chemins_verre = lister_images(os.path.join(dataset_dir, "verre"))
    chemins_metal = lister_images(os.path.join(dataset_dir, "metal"))
    chemins = chemins_plastic + chemins_verre + chemins_metal
    classes = np.array([0] * len(chemins_plastic) + [1] * len(chemins_verre) + [2] * len(chemins_metal),
                       dtype=np.int32)
    vecteurs = np.ascontiguousarray([image_vers_vecteur(c) for c in chemins], dtype=np.float64)

    dossier_modeles = os.path.join(DOSSIER, "modeles")
    os.makedirs(dossier_modeles, exist_ok=True)

    for classe_id, nom in NOMS_CLASSES.items():
        labels = np.where(classes == classe_id, 1, -1).astype(np.int32)
        coefs = np.zeros(len(vecteurs))
        n_supports = lib.svm_entrainer(
            vecteurs.ctypes.data_as(P_DOUBLE), labels.ctypes.data_as(P_INT),
            len(vecteurs), vecteurs.shape[1], RBF, 1.0, 1000.0, 10000,
            coefs.ctypes.data_as(P_DOUBLE))

        chemin = os.path.join(dossier_modeles, f"svm_{nom}.txt")
        lib.svm_sauvegarder(chemin.encode(), vecteurs.ctypes.data_as(P_DOUBLE),
                            coefs.ctypes.data_as(P_DOUBLE), len(vecteurs), vecteurs.shape[1], RBF, 1.0)
        print(f"{nom} : {n_supports} vecteurs de support sur {len(vecteurs)} photos, sauvegarde dans {chemin}")


if __name__ == "__main__":
    main()
