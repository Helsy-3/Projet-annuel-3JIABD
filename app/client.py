import sys
import numpy as np
import requests
from PIL import Image
from pathlib import Path

# Adresse du serveur Flask
URL = "http://127.0.0.1:5000"

# Désactiver l'utilisation des proxys système
session = requests.Session()
session.trust_env = False


# --------------------------------------------------
# 1. TRANSFORMER UNE IMAGE EN VECTEUR
# --------------------------------------------------

def image_vers_vecteur(chemin, taille=(2, 2)):

    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(f"Image introuvable : {chemin}")

    # Ouvrir l'image et la convertir en RGB
    with Image.open(chemin) as image:
        img = image.convert("RGB")
        img = img.resize(taille)

        # Normaliser les pixels entre 0 et 1
        vecteur = (
            np.array(img, dtype=np.float64) / 255.0
        ).flatten().tolist()

    return vecteur


# --------------------------------------------------
# 2. COMMUNIQUER AVEC LE SERVEUR ET PREDIRE
# --------------------------------------------------

def main():

    # Vérifier que le serveur fonctionne
    try:
        reponse = session.get(f"{URL}/ping", timeout=5)
        reponse.raise_for_status()

        etat = reponse.json()

        print("Connexion au serveur :", URL)
        print("Endpoint de vérification :", f"{URL}/ping")
        print("État du serveur :", etat)

        if not etat.get("modeles_charges"):
            print("Erreur : aucun modèle RBF n'est chargé.")
            return

    except requests.exceptions.RequestException as e:
        print("Impossible de contacter le serveur Flask.")
        print("Vérifie qu'il est bien démarré.")
        print("Détail :", e)
        return

    # Vérifier qu'une image a été fournie
    if len(sys.argv) < 2:
        print("Utilisation :")
        print("python3 app/client.py chemin/vers/image.jpg")
        return

    chemin_photo = sys.argv[1]

    try:

        # Transformer l'image en vecteur
        vecteur = image_vers_vecteur(chemin_photo)

        print("Image analysée :", chemin_photo)
        print("Nombre de caractéristiques :", len(vecteur))
        reponse = session.post(
        f"{URL}/predict",
        json={"vecteur": vecteur},
        timeout=10
        )

        # Envoyer le vecteur au serveur Flask
        reponse = session.post(
            f"{URL}/predict",
            json={"vecteur": vecteur},
            timeout=10
        )

        reponse.raise_for_status()

        resultat = reponse.json()

        # Afficher le résultat
        print("\nRésultat de la prédiction :")
        print("Classe prédite :", resultat["classe"])

        print("\nScores par classe :")

        for classe, score in resultat["scores"].items():
            print(f"{classe} : {score:.6f}")

    except FileNotFoundError as e:
        print("Erreur :", e)

    except requests.exceptions.HTTPError as e:
        print("Erreur HTTP :", e)
        try:
            print("Détail :", reponse.json())
        except ValueError:
            pass

    except requests.exceptions.RequestException as e:
        print("Erreur lors de la requête :", e)

    except Exception as e:
        print("Erreur :", e)


# --------------------------------------------------
# 3. POINT D'ENTREE DU PROGRAMME
# --------------------------------------------------

if __name__ == "__main__":
    main()