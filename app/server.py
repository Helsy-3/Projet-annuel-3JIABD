import ctypes
import os

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)

lib = ctypes.CDLL(os.path.join(RACINE, "C-algorithms", "ETAPE4.svm.so"))

P_DOUBLE = ctypes.POINTER(ctypes.c_double)

lib.svm_lire_entete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
lib.svm_lire_entete.restype = ctypes.c_int
lib.svm_charger.argtypes = [ctypes.c_char_p, P_DOUBLE, P_DOUBLE]
lib.svm_charger.restype = ctypes.c_int
lib.svm_score.argtypes = [P_DOUBLE, P_DOUBLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, P_DOUBLE]
lib.svm_score.restype = ctypes.c_double

NOMS_CLASSES = ["plastique", "verre", "metal"]


def charger_modele(nom):
    """Charge un modele SVM sauvegarde par app/entrainer_modele.py (un contre tous, un fichier par classe)."""
    chemin = os.path.join(DOSSIER, "modeles", f"svm_{nom}.txt").encode()
    n, d, noyau = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    gamma = ctypes.c_double()
    if lib.svm_lire_entete(chemin, n, d, noyau, gamma) != 0:
        return None
    X = np.zeros((n.value, d.value))
    coefs = np.zeros(n.value)
    lib.svm_charger(chemin, X.ctypes.data_as(P_DOUBLE), coefs.ctypes.data_as(P_DOUBLE))
    return {"X": X, "coefs": coefs, "noyau": noyau.value, "gamma": gamma.value}


# On charge les modeles une seule fois, au demarrage du serveur (pas a chaque requete).
MODELES = {nom: m for nom in NOMS_CLASSES if (m := charger_modele(nom)) is not None}


@app.route("/ping")
def ping():
    return {"message": "le serveur fonctionne"}


@app.route("/predict", methods=["POST"])
def predict():
    """Attend un JSON {"vecteur": [12 nombres entre 0 et 1]} (une photo deja reduite en
    2x2 pixels et normalisee, comme image_vers_vecteur() le fait cote client) et renvoie
    la classe predite par le modele SVM sauvegarde."""
    if not MODELES:
        return jsonify({"erreur": "aucun modele charge - lance d'abord python app/entrainer_modele.py"}), 503

    vecteur = np.ascontiguousarray(request.get_json()["vecteur"], dtype=np.float64)

    scores = {}
    for nom, m in MODELES.items():
        scores[nom] = lib.svm_score(m["X"].ctypes.data_as(P_DOUBLE), m["coefs"].ctypes.data_as(P_DOUBLE),
                                    len(m["X"]), m["X"].shape[1], m["noyau"], m["gamma"],
                                    vecteur.ctypes.data_as(P_DOUBLE))

    classe = max(scores, key=scores.get)
    return jsonify({"classe": classe, "scores": scores})


if __name__ == "__main__":
    app.run(port=5000)
