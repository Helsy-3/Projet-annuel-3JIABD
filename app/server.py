import ctypes
import os
import numpy as np
from flask import Flask, render_template, jsonify, request
import io
from PIL import Image

app = Flask(__name__)

# --------------------------------------------------
# 1. TROUVER LES FICHIERS DU PROJET
# --------------------------------------------------

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)

CHEMIN_LIB_RBF = os.path.join(
    RACINE, "C-algorithms", "ETAPE5.RBF.so"
)
CHEMIN_LIB_PMC = os.path.join(
    RACINE, "C-algorithms", "ETAPE3.PMC.so"
)

DOSSIER_MODELES = os.path.join(DOSSIER, "modeles")

NOMS_CLASSES = ["plastique", "verre", "metal"]

# Type utilisé pour envoyer des tableaux de nombres à la bibliothèque C.
P_DOUBLE = ctypes.POINTER(ctypes.c_double)


# --------------------------------------------------
# CHARGER RBF LIBRARY
# --------------------------------------------------

lib_rbf = ctypes.CDLL(CHEMIN_LIB_RBF)

# Décrire les arguments de la fonction C rbf_charger.
lib_rbf.rbf_charger.argtypes = [
    ctypes.c_char_p,
    P_DOUBLE,
    P_DOUBLE,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    P_DOUBLE
]
lib_rbf.rbf_charger.restype = ctypes.c_int

# Décrire les arguments de la fonction qui calcule les activations RBF.
lib_rbf.compute_dot_proximity_to_centroids.argtypes = [
    ctypes.c_int,
    P_DOUBLE,
    ctypes.c_int,
    P_DOUBLE,
    ctypes.c_double,
    P_DOUBLE
]

lib_rbf.compute_dot_proximity_to_centroids.restype = None

# Décrire les arguments de la fonction qui calcule le score final.
# Le nom doit correspondre à celui défini dans ton fichier C.
lib_rbf.calculer_dot_output.argtypes = [
    ctypes.c_int,
    P_DOUBLE,
    P_DOUBLE
]

lib_rbf.calculer_dot_output.restype = ctypes.c_double


# --------------------------------------------------
# CHARGER PMC LIBRARY
# --------------------------------------------------

lib_pmc = ctypes.CDLL(CHEMIN_LIB_PMC)

lib_pmc.pmc_charger.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(P_DOUBLE),
    ctypes.POINTER(P_DOUBLE)
]

lib_pmc.pmc_charger.restype = ctypes.c_int

lib_pmc.pmc_predire.argtypes = [
    P_DOUBLE,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(P_DOUBLE),
    ctypes.POINTER(P_DOUBLE),
    P_DOUBLE
]

lib_pmc.pmc_predire.restype = ctypes.c_int


# --------------------------------------------------
# CHARGER LE MODELE PMC
# --------------------------------------------------

def charger_modele_pmc():
    chemin = os.path.join(DOSSIER_MODELES, "pmc_weights.txt")
    if not os.path.isfile(chemin):
        print("Fichier PMC introuvable :", chemin)
        return None

    with open(chemin, "r", encoding="utf-8") as fichier:
        lignes = fichier.readlines()
    nb_couches = int(lignes[0].strip())
    dimensions = list(map(int, lignes[1].split()))

    if len(dimensions) != nb_couches:
        raise ValueError("Dimensions PMC incorrectes.")

    if dimensions[0] != 12 or dimensions[-1] != len(NOMS_CLASSES):
        raise ValueError("Architecture PMC incompatible.")

    weights = []
    biases = []

    for i in range(nb_couches - 1):
        weights.append(
            np.zeros(dimensions[i] * dimensions[i + 1], dtype=np.float64)
        )
        biases.append(
            np.zeros(dimensions[i + 1], dtype=np.float64)
        )

    weights_ptrs = (P_DOUBLE * (nb_couches - 1))()
    biases_ptrs = (P_DOUBLE * (nb_couches - 1))()

    for i in range(nb_couches - 1):
        weights_ptrs[i] = weights[i].ctypes.data_as(P_DOUBLE)
        biases_ptrs[i] = biases[i].ctypes.data_as(P_DOUBLE)

    dimensions_c = (ctypes.c_int * nb_couches)(*dimensions)

    resultat = lib_pmc.pmc_charger(
        os.fsencode(chemin),
        nb_couches,
        dimensions_c,
        weights_ptrs,
        biases_ptrs
    )
    if resultat != 0:
        raise ValueError("Impossible de charger le modèle PMC.")

    print("Modèle PMC chargé :", chemin)
    print("Architecture PMC :", dimensions)

    return {
        "nb_couches": nb_couches,
        "dimensions": dimensions,
        "dimensions_c": dimensions_c,
        "weights": weights,
        "biases": biases,
        "weights_ptrs": weights_ptrs,
        "biases_ptrs": biases_ptrs
    }

MODELE_PMC = charger_modele_pmc()


# --------------------------------------------------
# CALCULER LES SCORES DU PMC
# --------------------------------------------------

def calculer_scores_pmc(vecteur):

    if MODELE_PMC is None:
        raise ValueError("Le modèle PMC n'est pas chargé.")

    vecteur = np.ascontiguousarray(vecteur, dtype=np.float64)

    dimensions = MODELE_PMC["dimensions"]
    nb_couches = MODELE_PMC["nb_couches"]

    if vecteur.ndim != 1 or len(vecteur) != dimensions[0]:
        raise ValueError("Le PMC attend 12 valeurs en entrée.")

    if not np.isfinite(vecteur).all():
        raise ValueError("Le vecteur contient des valeurs invalides.")

    sorties = np.zeros(dimensions[-1], dtype=np.float64)

    resultat = lib_pmc.pmc_predire(
        vecteur.ctypes.data_as(P_DOUBLE),
        nb_couches,
        MODELE_PMC["dimensions_c"],
        MODELE_PMC["weights_ptrs"],
        MODELE_PMC["biases_ptrs"],
        sorties.ctypes.data_as(P_DOUBLE)
    )

    if resultat != 0:
        raise ValueError("Erreur pendant la prédiction PMC.")

    return {
        NOMS_CLASSES[i]: float(sorties[i])
        for i in range(len(NOMS_CLASSES))
    }

# --------------------------------------------------
# 3. CHARGER UN MODELE RBF
# --------------------------------------------------

def charger_modele_rbf(nom):

    # Exemple : app/modeles/rbf_plastique.txt
    chemin = os.path.join(
        DOSSIER_MODELES, "rbf_" + nom + ".txt"
    )

    # Vérifier que le fichier existe.
    if not os.path.isfile(chemin):
        return None

    # Lire la première ligne : K, dimensions et sigma.
    with open(chemin, "r", encoding="utf-8") as fichier:
        ligne = fichier.readline()

    informations = ligne.split()

    if len(informations) != 3:
        raise ValueError("En-tête incorrect : " + chemin)

    k = int(informations[0])
    dimensions = int(informations[1])
    sigma = float(informations[2])

    # Créer les tableaux qui recevront les paramètres.
    centroids = np.zeros((k, dimensions), dtype=np.float64)
    poids = np.zeros(k, dtype=np.float64)

    # Variables que la fonction C va remplir.
    k_charge = ctypes.c_int()
    dimensions_charge = ctypes.c_int()
    sigma_charge = ctypes.c_double()

    # Charger les paramètres enregistrés dans le fichier.
    resultat = lib_rbf.rbf_charger(
        os.fsencode(chemin),
        centroids.ctypes.data_as(P_DOUBLE),
        poids.ctypes.data_as(P_DOUBLE),
        ctypes.byref(k_charge),
        ctypes.byref(dimensions_charge),
        ctypes.byref(sigma_charge)
    )

    if resultat != 0:
        raise ValueError("Impossible de charger : " + chemin)

    print("Fichier testé :", chemin)
    print("Résultat du chargement C :", resultat)
    print("K, dimensions, sigma :", k_charge.value, dimensions_charge.value, sigma_charge.value)

    # Retourner tous les paramètres du modèle.
    modele = {
        "centroids": centroids,
        "poids": poids,
        "k": k,
        "dimensions": dimensions,
        "sigma": sigma
    }

    return modele


# --------------------------------------------------
# 4. CHARGER LES TROIS MODELES AU DEMARRAGE
# --------------------------------------------------

MODELES = {}

for nom in NOMS_CLASSES:
    modele = charger_modele_rbf(nom)

    if modele is not None:
        MODELES[nom] = modele


# --------------------------------------------------
# 5. CALCULER LE SCORE D'UN MODELE
# --------------------------------------------------

def calculer_score(modele, vecteur):

    # Convertir le vecteur en tableau de nombres compatible avec C.
    vecteur = np.ascontiguousarray(vecteur, dtype=np.float64)

    k = modele["k"]
    dimensions = modele["dimensions"]
    
    # Tableau qui contiendra la proximité avec chaque centroïde.
    proximites = np.zeros(k, dtype=np.float64)

    # Calculer les activations RBF grâce à la fonction C.
    lib_rbf.compute_dot_proximity_to_centroids(
        dimensions,
        vecteur.ctypes.data_as(P_DOUBLE),
        k,
        modele["centroids"].ctypes.data_as(P_DOUBLE),
        modele["sigma"],
        proximites.ctypes.data_as(P_DOUBLE)
    )

    # Combiner les activations avec les poids appris.
    score = lib_rbf.calculer_dot_output(
        k,
        proximites.ctypes.data_as(P_DOUBLE),
        modele["poids"].ctypes.data_as(P_DOUBLE)
    )

    return float(score)


# --------------------------------------------------
# 6. VERIFIER QUE LE SERVEUR FONCTIONNE
# --------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():

    return jsonify({
        "message": "Le serveur RBF fonctionne",
        "modeles_charges": list(MODELES.keys())
    })


# --------------------------------------------------
# 7. PREDIRE LA CLASSE D'UN VECTEUR
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Vérifier que des modèles ont été chargés.
    if len(MODELES) == 0:
        return jsonify({
            "erreur": "Aucun modèle chargé. Entraîne les modèles avant."
        }), 503

    # Récupérer les données envoyées par le client.
    donnees = request.get_json(silent=True)

    if not isinstance(donnees, dict):
        return jsonify({
            "erreur": "La requête doit contenir un objet JSON."
        }), 400

    if "vecteur" not in donnees:
        return jsonify({
            "erreur": "Le champ vecteur est obligatoire."
        }), 400

    # Transformer les données reçues en nombres.
    try:
        vecteur = np.asarray(
            donnees["vecteur"], dtype=np.float64
        )
    except (TypeError, ValueError):
        return jsonify({
            "erreur": "Le vecteur doit contenir des nombres."
        }), 400

    # Vérifier que le vecteur est une liste de nombres valide.
    if vecteur.ndim != 1 or len(vecteur) == 0:
        return jsonify({
            "erreur": "Le vecteur doit être une liste non vide."
        }), 400

    if not np.isfinite(vecteur).all():
        return jsonify({
            "erreur": "Le vecteur contient des valeurs invalides."
        }), 400

    # Vérifier le nombre de caractéristiques attendu.
    premier_modele = next(iter(MODELES.values()))
    dimensions = premier_modele["dimensions"]

    if len(vecteur) != dimensions:
        return jsonify({
            "erreur": "Le vecteur doit contenir "
                      + str(dimensions) + " valeurs."
        }), 400

    # Calculer le score de chaque classe.
    scores = {}

    for nom in MODELES:
        modele = MODELES[nom]
        score = calculer_score(modele, vecteur)
        scores[nom] = score

    # Choisir la classe qui possède le score le plus élevé.
    classe = max(scores, key=scores.get)

    # Envoyer la prédiction au client.
    return jsonify({
        "classe": classe,
        "scores": scores
    })


# --------------------------------------------------
# 8. DEMARRER LE SERVEUR
# --------------------------------------------------

# PAGE PRINCIPALE DU SITE
@app.route("/")
def accueil():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION PMC A PARTIR D'UN VECTEUR
# --------------------------------------------------
@app.route("/predict-pmc-vecteur", methods=["POST"])
def predict_pmc():

    if MODELE_PMC is None:
        return jsonify({"erreur": "Modèle PMC non chargé"}), 503

    donnees = request.get_json(silent=True)

    if not isinstance(donnees, dict) or "vecteur" not in donnees:
        return jsonify({"erreur": "Le champ vecteur est obligatoire"}), 400

    try:
        vecteur = np.asarray(donnees["vecteur"], dtype=np.float64)

        scores = calculer_scores_pmc(vecteur)
        classe = max(scores, key=scores.get)

        return jsonify({
            "modele": "PMC",
            "classe": classe,
            "scores": scores
        })

    except (TypeError, ValueError) as e:
        return jsonify({"erreur": str(e)}), 400


# --------------------------------------------------
# PREDICTION PMC A PARTIR D'UNE IMAGE
# --------------------------------------------------
@app.route("/predict-image-pmc", methods=["POST"])
def predict_image_pmc():

    if MODELE_PMC is None:
        return jsonify({"erreur": "Modèle PMC non chargé"}), 503

    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue"}), 400

    try:
        fichier = request.files["image"]

        image = Image.open(
            io.BytesIO(fichier.read())
        ).convert("RGB")

        image = image.resize((2, 2))

        vecteur = (
            np.asarray(image, dtype=np.float64) / 255.0
        ).flatten()

        scores = calculer_scores_pmc(vecteur)
        classe = max(scores, key=scores.get)

        return jsonify({
            "modele": "PMC",
            "classe": classe,
            "scores": scores
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 400

    

# --------------------------------------------------
# PREDICTION RBF A PARTIR D'UNE IMAGE
# --------------------------------------------------
@app.route("/predict-image-rbf", methods=["POST"])
def predict_image():

    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue"}), 400

    try:
        fichier = request.files["image"]

        # Transformer l'image en vecteur de 12 valeurs
        image = Image.open(io.BytesIO(fichier.read())).convert("RGB")
        image = image.resize((2, 2))

        vecteur = (
            np.asarray(image, dtype=np.float64) / 255.0
        ).flatten()

        # Calculer les scores des trois modèles
        scores = {}

        for nom, modele in MODELES.items():
            scores[nom] = calculer_score(modele, vecteur)

        # Choisir la classe ayant le meilleur score
        classe = max(scores, key=scores.get)

        return jsonify({
            "classe": classe,
            "scores": scores
        })

    except Exception as e:
        return jsonify({"erreur": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)