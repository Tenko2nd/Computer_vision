# Instructions d'utilisation

Ce document décrit comment utiliser les scripts Python fournis pour obtenir les paramètres de la caméra et visualiser les simulations vidéo.

## Prérequis et Installation

**Installer les dépendances :**

installez les dépendances listées dans `requirements.txt` .

```bash
pip install -r requirements.txt
```

## 1. Obtenir les paramètres de la caméra

Pour récupérer les paramètres intrinsèques de la caméra :

*   Exécutez le script Python `Camera_parameter.py`.

    ```bash
    python Camera_parameter.py
    ```

    *Assurez-vous que les dépendances nécessaires sont installées et que les images/données requises par le script sont disponibles.*

## 2. Voir les simulations vidéo (Homographie)

Pour visualiser les simulations vidéo basées sur l'homographie :

1.  **Configurer la vidéo :**
    *   Ouvrez le fichier `homographie.py` dans un éditeur de texte.
    *   Localisez la ligne où la variable `vid_name` est définie.
    *   Modifiez la valeur de `vid_name` pour qu'elle corresponde au nom (ou au chemin d'accès complet) du fichier vidéo que vous souhaitez utiliser pour la simulation.

    ```python
    if __name__ == '__main__':
        # choisir un nom de video pour la selectionner (parmis ['mousse', 'rugby', 'tennis'])
        vid_name = 'tennis'
    ```

2.  **Lancer la simulation :**
    *   Exécutez le script Python `homographie.py`.

        ```bash
        python homographie.py
        ```

    *Le script traitera la vidéo spécifiée par `vid_name` et affichera la simulation en temps réel.*