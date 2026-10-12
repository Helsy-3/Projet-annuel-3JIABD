""" api calls to retrieve big dataset """
import os
import time
import requests


API_KEY = os.getenv("PEXELS_API_KEY")
BASE_URL = "https://api.pexels.com/v1/search"

HEADERS = {
    "Authorization": API_KEY
}


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


if __name__ == "__main__":

    if not API_KEY:
        raise ValueError(
            "Définis ta clé API dans la variable d'environnement PEXELS_API_KEY"
        )

    images = get_bottle_images(
        query="glass bottle",
        per_page=80,
        max_pages=10
    )

    print(f"\nTotal : {len(images)} images")

    import json

    with open("bottle_images.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)


def get_glass_images():


def get_metal_images():