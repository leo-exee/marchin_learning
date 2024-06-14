## Prérequis
Avant de pouvoir utiliser projet, veillez à créer le venv avec Python (version : `3.12.0`) et d'installer les librairies :
```bash
pip install -r requirements.txt #pip3 si pip ne fonctionne pas
```

## Objectif
L'objectif du projet est de créer un système capable de noter des commentaire en tant que positif, négatif ou neutre. En fonction des données fournie dans le `data.csv`, le programme peut être entrainer à identifier l'émontion d'un texte, puis directement l'appliquer avec un texte donné.

## Utilisation

1. Prétraitez les données :

```bash
python src/data_preprocessing.py
```

2. Entraînez le modèle :

```bash
python src/model_training.py
```

3. Évaluez le modèle :

```bash
python src/model_evaluation.py
```

4. Prédisez de nouveaux commentaires :

```bash
python src/predict.py
```
