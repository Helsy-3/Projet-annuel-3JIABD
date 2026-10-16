import sys

import numpy as np
import requests
from PIL import Image

URL = "http://127.0.0.1:5000"


def image_vers_vecteur(chemin, taille=(2, 2)):
    img = Image.open(chemin).convert("RGB")
    img = img.resize(taille)
    return (np.array(img) / 255.0).flatten().tolist()


reponse = requests.get(f"{URL}/ping")
print(reponse.json())

if len(sys.argv) > 1:
    chemin_photo = sys.argv[1]
    vecteur = image_vers_vecteur(chemin_photo)
    reponse = requests.post(f"{URL}/predict", json={"vecteur": vecteur})
    print(reponse.json())
else:
    print("Astuce : python app/client.py chemin/vers/une/photo.jpg pour demander une prediction")
