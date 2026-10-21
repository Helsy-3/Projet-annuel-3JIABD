"""
entrainer_modele_rbf.py - Entraine le RBF Network (ETAPE5.RBNF.c) sur le dataset reel
et sauvegarde un modele par classe (un contre tous) dans app/modeles/, pour que
server.py puisse les charger et predire sur de nouvelles photos, sans avoir a tout
ré-entrainer a chaque fois.

Utilisation (depuis la racine du repo) :
    python app/entrainer_modele_rbf.py
"""

import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
sys.path.insert(0, os.path.join(RACINE, "C-algorithms"))

from test_etape5 import RBF, image_vers_vecteur, lister_images  # reutilise la classe deja testee

NOMS_CLASSES = {0: "plastique", 1: "verre", 2: "metal"}


def main():
    dataset_dir = os.path.join(RACINE, "dataset-img")
    chemins_plastic = lister_images(os.path.join(dataset_dir, "plastic"))
    chemins_verre = lister_images(os.path.join(dataset_dir, "verre"))
    chemins_metal = lister_images(os.path.join(dataset_dir, "metal"))
    chemins = chemins_plastic + chemins_verre + chemins_metal
    classes = np.array([0] * len(chemins_plastic) + [1] * len(chemins_verre) + [2] * len(chemins_metal))
    vecteurs = np.array([image_vers_vecteur(c) for c in chemins], dtype=np.float64)

    dossier_modeles = os.path.join(DOSSIER, "modeles")
    os.makedirs(dossier_modeles, exist_ok=True)

    for classe_id, nom in NOMS_CLASSES.items():
        labels = np.where(classes == classe_id, 1, -1).astype(float)
        modele = RBF(k=8).entrainer(vecteurs, labels)

        chemin = os.path.join(dossier_modeles, f"rbf_{nom}.txt")
        modele.sauvegarder(chemin)
        precision = (modele.predire(vecteurs) == labels).mean()
        print(f"{nom} : {modele.k} centroides, precision entrainement {precision:.0%}, sauvegarde dans {chemin}")


if __name__ == "__main__":
    main()
