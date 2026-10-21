"""
test_sous_sur_apprentissage.py - Met en evidence le sous-apprentissage et le
sur-apprentissage sur le dataset reel (photos de dechets), en faisant varier la
"complexite" de chaque modele :
  - Perceptron  : nombre d'epochs d'entrainement (max_iter)
  - PMC         : taille de la couche cachee (h_dim)
  - SVM         : gamma du noyau RBF (largeur de la gaussienne)
  - RBF Network : nombre de clusters (K)

Pour chaque valeur de complexite, on entraine UNIQUEMENT sur les photos
d'entrainement, puis on mesure la precision a la fois sur l'entrainement et sur
le test (jamais vu), et on trace les deux courbes l'une a cote de l'autre.

Ce qu'on doit observer (les phenomenes du cours) :
  - Complexite trop faible : les deux courbes sont basses et proches l'une de
    l'autre -> SOUS-APPRENTISSAGE, le modele est trop simple pour apprendre le
    probleme, meme sur les donnees qu'il a deja vues.
  - Complexite trop grande : la courbe d'entrainement monte pres de 100% mais la
    courbe de test stagne ou redescend -> SUR-APPRENTISSAGE, le modele a appris
    les photos d'entrainement "par coeur" (y compris leur bruit) au lieu
    d'apprendre une regle generale qui marche aussi sur de nouvelles photos.
  - Entre les deux se trouve le "bon" niveau de complexite : assez riche pour
    bien apprendre, assez simple pour generaliser.

En plus de ces courbes de COMPLEXITE (figure 1), ce script trace aussi des courbes
d'APPRENTISSAGE (figure 2) : a complexite FIXE cette fois, on regarde comment la
precision evolue au fil des EPOCHS d'entrainement. Ca montre le moment ou continuer
a entrainer plus longtemps commence a nuire a la generalisation (la precision de
test plafonne ou redescend pendant que celle d'entrainement continue de monter).

Utilisation (depuis la racine du repo), apres avoir compile les bibliotheques :
    python C-algorithms/test_sous_sur_apprentissage.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DOSSIER)

import test_etape1 as e1  # Perceptron
import test_etape3 as e3  # PMC (entrainer_pmc/predire_pmc, h_dim variable)
import test_etape4 as e4  # SVM (gamma variable)
import test_etape5 as e5  # RBF Network (K variable), + helpers dataset reel


def charger_dataset_reel():
    chemins_plastic = e5.lister_images(os.path.join(DOSSIER, "..", "dataset-img", "plastic"))
    chemins_verre = e5.lister_images(os.path.join(DOSSIER, "..", "dataset-img", "verre"))
    chemins_metal = e5.lister_images(os.path.join(DOSSIER, "..", "dataset-img", "metal"))
    chemins = chemins_plastic + chemins_verre + chemins_metal
    classes = np.array([0] * len(chemins_plastic) + [1] * len(chemins_verre) + [2] * len(chemins_metal))

    chemins_train, classes_train, chemins_test, classes_test = e5.separer_train_test(chemins, classes)
    vecteurs_train = np.array([e5.image_vers_vecteur(c) for c in chemins_train], dtype=np.float64)
    vecteurs_test = np.array([e5.image_vers_vecteur(c) for c in chemins_test], dtype=np.float64)
    return vecteurs_train, classes_train, vecteurs_test, classes_test


def courbe_perceptron(X_train, y_train, X_test, y_test):
    """Le perceptron lineaire a une complexite fixe (un seul neurone) : peu importe le
    nombre d'epochs, il ne peut pas apprendre une frontiere non lineaire. On s'attend a
    ce que les deux courbes plafonnent bas, tout de suite - un sous-apprentissage
    structurel, pas un manque d'entrainement."""
    # IMPORTANT : on melange les donnees avant l'entrainement. Sans ca, comme le dataset
    # est range par classe (tout le plastique, puis tout le verre, puis tout le metal),
    # le perceptron (qui apprend point par point, dans l'ordre) est biaise vers la
    # DERNIERE classe rencontree des qu'on limite le nombre d'epochs - un exemple concret
    # de biais introduit par l'ordre des donnees, pas par l'algorithme lui-meme.
    rng_melange = np.random.default_rng(0)
    ordre = rng_melange.permutation(len(X_train))
    X_train_melange, y_train_melange = X_train[ordre], y_train[ordre]

    valeurs = [1, 5, 10, 50, 100, 500, 1000]
    precision_train, precision_test = [], []
    for max_iter in valeurs:
        modele = e1.PerceptronMultiClasse(max_iter=max_iter).entrainer(X_train_melange, y_train_melange)
        precision_train.append((modele.predire(X_train) == y_train).mean())
        precision_test.append((modele.predire(X_test) == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_biais_ordre_donnees(X_train, y_train, X_test, y_test):
    """Demontre un biais concret du dataset : nos photos sont rangees par classe (tout
    le plastique, puis tout le verre, puis tout le metal). Le perceptron apprend point
    par point, dans l'ordre - avec peu d'epochs, il n'a pas le temps de corriger ce
    biais et se retrouve pousse vers la DERNIERE classe rencontree (metal), peu importe
    la vraie classe du point. On compare les memes entrainements, avec et sans melange
    prealable des donnees."""
    rng_melange = np.random.default_rng(0)
    ordre = rng_melange.permutation(len(X_train))
    X_melange, y_melange = X_train[ordre], y_train[ordre]

    valeurs = [1, 5, 10, 50, 100, 500, 1000]
    precision_sans_melange, precision_avec_melange = [], []
    for max_iter in valeurs:
        m_sans = e1.PerceptronMultiClasse(max_iter=max_iter).entrainer(X_train, y_train)
        m_avec = e1.PerceptronMultiClasse(max_iter=max_iter).entrainer(X_melange, y_melange)
        precision_sans_melange.append((m_sans.predire(X_test) == y_test).mean())
        precision_avec_melange.append((m_avec.predire(X_test) == y_test).mean())
    return valeurs, precision_sans_melange, precision_avec_melange


def courbe_pmc(X_train, y_train, X_test, y_test):
    """h_dim = nombre de neurones caches = la "complexite" du PMC. Avec tres peu de
    neurones, il ne peut pas apprendre grand-chose (sous-apprentissage). Avec beaucoup
    de neurones et tres peu de photos (65 en entrainement), il peut finir par apprendre
    les photos par coeur (sur-apprentissage)."""
    valeurs = [1, 2, 4, 8, 16, 32]
    precision_train, precision_test = [], []
    for h_dim in valeurs:
        bonnes_train, poids = e3.entrainer_pmc(X_train, y_train, h_dim=h_dim)
        predictions_test = np.array([e3.predire_pmc(v, poids) for v in X_test])
        precision_train.append(bonnes_train / len(X_train))
        precision_test.append((predictions_test == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_svm(X_train, y_train, X_test, y_test):
    """gamma = a quel point le noyau RBF est "pointu". Un petit gamma donne une
    frontiere tres lisse (sous-apprentissage si trop petit). Un grand gamma colle
    exactement a chaque point d'entrainement (sur-apprentissage : ca "memorise" les
    photos au lieu d'apprendre une regle generale)."""
    valeurs = [0.01, 0.1, 0.5, 1, 2, 5, 10, 50]
    precision_train, precision_test = [], []
    for gamma in valeurs:
        modele = e4.SVMMultiClasse(noyau=e4.RBF, gamma=gamma).entrainer(X_train, y_train)
        precision_train.append((modele.predire(X_train) == y_train).mean())
        precision_test.append((modele.predire(X_test) == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_rbf(X_train, y_train, X_test, y_test):
    """K = nombre de centroides. Peu de centroides = le RBF ne peut capturer que des
    formes tres grossieres (sous-apprentissage). Beaucoup de centroides par rapport
    au nombre de photos = un centroide presque par photo, le modele "memorise"
    chaque exemple d'entrainement individuellement (sur-apprentissage)."""
    valeurs = [2, 4, 6, 10, 15, 25, 40]
    precision_train, precision_test = [], []
    for k in valeurs:
        with e5.silencieux():
            modele = e5.RBFMultiClasse(k=k).entrainer(X_train, y_train)
        precision_train.append((modele.predire(X_train) == y_train).mean())
        precision_test.append((modele.predire(X_test) == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_apprentissage_pmc(X_train, y_train, X_test, y_test, h_dim=8):
    """Complexite FIXE (h_dim), on fait seulement varier le nombre d'epochs."""
    valeurs = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    precision_train, precision_test = [], []
    for epochs in valeurs:
        bonnes_train, poids = e3.entrainer_pmc(X_train, y_train, h_dim=h_dim, epochs=epochs)
        predictions_test = np.array([e3.predire_pmc(v, poids) for v in X_test])
        precision_train.append(bonnes_train / len(X_train))
        precision_test.append((predictions_test == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_apprentissage_svm(X_train, y_train, X_test, y_test, gamma=1.0):
    """Complexite FIXE (gamma), on fait seulement varier max_iter. Le SVM part toujours
    des memes coefficients a 0 : ré-entrainer avec un max_iter plus grand donne
    exactement le meme resultat que si on avait continue l'entrainement precedent."""
    valeurs = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    precision_train, precision_test = [], []
    for max_iter in valeurs:
        modele = e4.SVMMultiClasse(noyau=e4.RBF, gamma=gamma, max_iter=max_iter).entrainer(X_train, y_train)
        precision_train.append((modele.predire(X_train) == y_train).mean())
        precision_test.append((modele.predire(X_test) == y_test).mean())
    return valeurs, precision_train, precision_test


def courbe_apprentissage_rbf(X_train, y_train, X_test, y_test, k=10):
    """Complexite FIXE (K) : on fait le clustering (k-means) UNE SEULE FOIS, partage par
    les 3 classes (le clustering ne depend que des photos, pas de leur classe), puis on
    ne fait varier que le nombre d'epochs d'entrainement des poids. Isole ainsi l'effet
    de la duree d'entrainement, sans le mélanger avec un nouveau tirage aleatoire de
    centroides a chaque fois."""
    X_train_c = np.ascontiguousarray(X_train, dtype=np.float64)
    n, d = X_train_c.shape
    k = max(2, min(k, n))

    modele_ref = e5.RBF(k=k)
    with e5.silencieux():
        modele_ref.entrainer(X_train_c, np.where(y_train == y_train[0], 1, -1).astype(float))
    centroids, sigma, k = modele_ref.centroids, modele_ref.sigma, modele_ref.k

    classes = np.unique(y_train)
    valeurs = [1, 5, 10, 50, 100, 300, 500, 1000]
    precision_train, precision_test = [], []
    for epochs in valeurs:
        modeles = []
        for c in classes:
            y_bin = np.where(y_train == c, 1, -1).astype(float)
            poids = np.zeros(k)
            with e5.silencieux():
                e5.lib.recursive_train_weights(k, n, d, X_train_c.ctypes.data_as(e5.P_DOUBLE),
                                     y_bin.ctypes.data_as(e5.P_DOUBLE),
                                     centroids.ctypes.data_as(e5.P_DOUBLE), sigma,
                                     poids.ctypes.data_as(e5.P_DOUBLE), 0.01, epochs)
            m = e5.RBF(k=k)
            m.k, m.centroids, m.sigma, m.poids = k, centroids, sigma, poids
            modeles.append(m)

        scores_train = np.array([m.score(X_train) for m in modeles])
        scores_test = np.array([m.score(X_test) for m in modeles])
        predictions_train = classes[np.argmax(scores_train, axis=0)]
        predictions_test = classes[np.argmax(scores_test, axis=0)]
        precision_train.append((predictions_train == y_train).mean())
        precision_test.append((predictions_test == y_test).mean())
    return valeurs, precision_train, precision_test


def tracer_courbe(ax, valeurs, precision_train, precision_test, titre, xlabel):
    ax.plot(valeurs, precision_train, marker="o", label="Entrainement", color="steelblue")
    ax.plot(valeurs, precision_test, marker="o", label="Test (jamais vu)", color="green")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title(titre, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    print("=== Sous-apprentissage / sur-apprentissage sur le dataset reel ===")
    X_train, y_train, X_test, y_test = charger_dataset_reel()
    print(f"Entrainement : {len(X_train)} photos, Test : {len(X_test)} photos (jamais vues)\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print("Perceptron (variation du nombre d'epochs)...")
    valeurs, p_train, p_test = courbe_perceptron(X_train, y_train, X_test, y_test)
    tracer_courbe(axes[0, 0], valeurs, p_train, p_test,
                 "Perceptron : complexite fixe (1 neurone)\n-> plafonne plus bas que les modeles non lineaires",
                 "Nombre d'epochs (max_iter)")

    print("PMC (variation de la taille de la couche cachee)...")
    valeurs, p_train, p_test = courbe_pmc(X_train, y_train, X_test, y_test)
    tracer_courbe(axes[0, 1], valeurs, p_train, p_test,
                 "PMC : precision vs taille de la couche cachee",
                 "Neurones caches (h_dim)")

    print("SVM (variation de gamma)...")
    valeurs, p_train, p_test = courbe_svm(X_train, y_train, X_test, y_test)
    tracer_courbe(axes[1, 0], valeurs, p_train, p_test,
                 "SVM (noyau RBF) : precision vs gamma",
                 "gamma")
    axes[1, 0].set_xscale("log")

    print("RBF Network (variation du nombre de clusters K)...")
    valeurs, p_train, p_test = courbe_rbf(X_train, y_train, X_test, y_test)
    tracer_courbe(axes[1, 1], valeurs, p_train, p_test,
                 "RBF Network : precision vs nombre de clusters",
                 "K (nombre de centroides)")

    fig.suptitle("Sous-apprentissage (gauche des courbes) vs sur-apprentissage (droite) - dataset reel",
                fontsize=12)
    fig.tight_layout()
    chemin = os.path.join(DOSSIER, "test_sous_sur_apprentissage.png")
    fig.savefig(chemin, dpi=110)
    print(f"\nFigure enregistree : {chemin}")

    print("\nBiais du dataset : perceptron avec/sans melange des donnees avant entrainement...")
    valeurs, sans, avec = courbe_biais_ordre_donnees(X_train, y_train, X_test, y_test)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(valeurs, sans, marker="o", label="Sans melanger (ordonne par classe)", color="tomato")
    ax2.plot(valeurs, avec, marker="o", label="Avec melange prealable", color="steelblue")
    ax2.set_xlabel("Nombre d'epochs (max_iter)")
    ax2.set_ylabel("Precision sur le test")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Biais du dataset : l'ordre des donnees fausse\nl'apprentissage avec peu d'epochs")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    chemin2 = os.path.join(DOSSIER, "test_biais_donnees.png")
    fig2.savefig(chemin2, dpi=110)
    print(f"Figure enregistree : {chemin2}")

    print("\n=== Courbes d'apprentissage (precision vs epochs, a complexite fixe) ===")
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    print("Perceptron...")
    valeurs, p_train, p_test = courbe_perceptron(X_train, y_train, X_test, y_test)
    tracer_courbe(axes3[0, 0], valeurs, p_train, p_test,
                 "Perceptron : precision vs epochs",
                 "Nombre d'epochs (max_iter)")

    print("PMC...")
    valeurs, p_train, p_test = courbe_apprentissage_pmc(X_train, y_train, X_test, y_test)
    tracer_courbe(axes3[0, 1], valeurs, p_train, p_test,
                 "PMC (h_dim=8 fixe) : precision vs epochs",
                 "Nombre d'epochs")
    axes3[0, 1].set_xscale("log")

    print("SVM...")
    valeurs, p_train, p_test = courbe_apprentissage_svm(X_train, y_train, X_test, y_test)
    tracer_courbe(axes3[1, 0], valeurs, p_train, p_test,
                 "SVM (gamma=1.0 fixe) : precision vs max_iter",
                 "max_iter")
    axes3[1, 0].set_xscale("log")

    print("RBF Network...")
    valeurs, p_train, p_test = courbe_apprentissage_rbf(X_train, y_train, X_test, y_test)
    tracer_courbe(axes3[1, 1], valeurs, p_train, p_test,
                 "RBF (K=10 fixe) : precision vs epochs",
                 "Nombre d'epochs")

    fig3.suptitle("Courbes d'apprentissage : precision en fonction de la duree d'entrainement\n"
                 "(complexite fixe cette fois - voir figure 1 pour la complexite)", fontsize=12)
    fig3.tight_layout()
    chemin3 = os.path.join(DOSSIER, "test_courbes_apprentissage.png")
    fig3.savefig(chemin3, dpi=110)
    print(f"\nFigure enregistree : {chemin3}")

    plt.show()


if __name__ == "__main__":
    main()
