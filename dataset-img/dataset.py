""" api calls to retrieve big dataset """
import os
import requests


API_KEY = os.getenv("PEXELS_API_KEY")
BASE_URL = "https://api.pexels.com/v1/search"

HEADERS = {
    "Authorization": API_KEY
}

DOSSIER = os.path.dirname(os.path.abspath(__file__))


def get_bottle_images(query="glass bottle", per_page=80, max_pages=10):
    images = []

    for page in range(1, max_pages + 1):

        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
        }

        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        photos = data.get("photos", [])

        if not photos:
            break

        for photo in photos:
            images.append({
                "id": photo["id"],
                "url": photo["url"],
                "image": photo["src"]["original"],
                "large": photo["src"]["large2x"],
                "medium": photo["src"]["medium"],
                "width": photo["width"],
                "height": photo["height"],
                "photographer": photo["photographer"],
            })

        print(f"Page {page}: {len(photos)} images récupérées")

    return images


def get_glass_images():
    return get_bottle_images(query="glass jar", per_page=80, max_pages=10)


def get_metal_images():
    return get_bottle_images(query="metal can", per_page=80, max_pages=10)


def get_plastic_images():
    return get_bottle_images(query="plastic bottle", per_page=80, max_pages=10)


def telecharger_images(images, categorie, proportion_test=0.2):
    """Telecharge chaque photo (a partir de son URL 'medium') et la range directement
    dans dataset-img/train/<categorie>/ ou dataset-img/test/<categorie>/ (~80%/20%,
    comme separer_train_test() ailleurs dans le projet) - pas besoin d'un script a part
    pour faire ce split, c'est fait ici au moment du telechargement."""
    n_test = max(1, round(len(images) * proportion_test))
    for i, image in enumerate(images, start=1):
        sous_dossier = "test" if i <= n_test else "train"
        dossier = os.path.join(DOSSIER, sous_dossier, categorie)
        os.makedirs(dossier, exist_ok=True)

        reponse = requests.get(image["medium"], timeout=30)
        reponse.raise_for_status()
        chemin = os.path.join(dossier, f"{i:02d}.jpg")
        with open(chemin, "wb") as f:
            f.write(reponse.content)
        print(f"Image {i}/{len(images)} telechargee : {chemin}")


if __name__ == "__main__":

    if not API_KEY:
        raise ValueError(
            "Définis ta clé API dans la variable d'environnement PEXELS_API_KEY"
        )

    categories = {
        "verre": get_glass_images,
        "metal": get_metal_images,
        "plastic": get_plastic_images,
    }

    resultats = {}
    for nom, fonction_recuperation in categories.items():
        print(f"\n=== {nom} ===")
        resultats[nom] = fonction_recuperation()
        print(f"Total : {len(resultats[nom])} images trouvees")

    # Classes equilibrees : on garde le meme nombre de photos par categorie (le plus
    # petit nombre trouve), sinon un modele peut avoir une bonne precision juste en
    # favorisant la classe la plus fournie, sans vraiment apprendre a distinguer les 3.
    n_min = min(len(images) for images in resultats.values())
    print(f"\nOn garde {n_min} photos par categorie pour avoir des classes equilibrees")

    for nom, images in resultats.items():
        telecharger_images(images[:n_min], nom)
