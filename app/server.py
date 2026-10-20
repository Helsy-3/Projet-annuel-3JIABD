import ctypes
import os
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# --------------------------------------------------
# 1. TROUVER LES FICHIERS DU PROJET
# --------------------------------------------------

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)

CHEMIN_LIB = os.path.join(
    RACINE, "C-algorithms", "ETAPE5.RBNF.so"
)

DOSSIER_MODELES = os.path.join(DOSSIER, "modeles")

NOMS_CLASSES = ["plastique", "verre", "metal"]

# Type utilisé pour envoyer des tableaux de nombres à la bibliothèque C.
P_DOUBLE = ctypes.POINTER(ctypes.c_double)


# --------------------------------------------------
# 2. CHARGER LA BIBLIOTHEQUE C
# --------------------------------------------------

lib = ctypes.CDLL(CHEMIN_LIB)

# Décrire les arguments de la fonction C rbf_charger.
lib.rbf_charger.argtypes = [
    ctypes.c_char_p,
    P_DOUBLE,
    P_DOUBLE,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    P_DOUBLE
]
lib.rbf_charger.restype = ctypes.c_int

# Décrire les arguments de la fonction qui calcule les activations RBF.
lib.calculer_rbf_closeness_to_centroid.argtypes = [
    ctypes.c_int,
    P_DOUBLE,
    ctypes.c_int,
    P_DOUBLE,
    ctypes.c_double,
    P_DOUBLE
]
lib.calculer_rbf_closeness_to_centroid.restype = None

# Décrire les arguments de la fonction qui calcule le score final.
# Le nom doit correspondre à celui défini dans ton fichier C.
lib.calculer_dot_output.argtypes = [
    ctypes.c_int,
    P_DOUBLE,
    P_DOUBLE
]
lib.calculer_dot_output.restype = ctypes.c_double


# --------------------------------------------------
# 3. CHARGER UN MODELE ENREGISTRE
# --------------------------------------------------

def charger_modele(nom):

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
    resultat = lib.rbf_charger(
        os.fsencode(chemin),
        centroids.ctypes.data_as(P_DOUBLE),
        poids.ctypes.data_as(P_DOUBLE),
        ctypes.byref(k_charge),
        ctypes.byref(dimensions_charge),
        ctypes.byref(sigma_charge)
    )

    if resultat != 0:
        raise ValueError("Impossible de charger : " + chemin)

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
    modele = charger_modele(nom)

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
    lib.calculer_rbf_closeness_to_centroid(
        dimensions,
        vecteur.ctypes.data_as(P_DOUBLE),
        k,
        modele["centroids"].ctypes.data_as(P_DOUBLE),
        modele["sigma"],
        proximites.ctypes.data_as(P_DOUBLE)
    )

    # Combiner les activations avec les poids appris.
    score = lib.calculer_dot_output(
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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)