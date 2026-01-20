# Prédiction du S&P500 avec un Réseau LSTM

**Projet Pédagogique - Deep Learning avec PyTorch**

Ce projet implémente un réseau de neurones récurrent LSTM (Long Short-Term Memory) pour prédire les cours de l'indice boursier S&P500.

---

## Table des matières

- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Scripts disponibles](#-scripts-disponibles)

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Clonez ou accédez au répertoire du projet** 
   

2. **Créez un environnement virtuel**  :
   ```bash
   python3 -m venv env
   source env/bin/activate  # Sur Linux/Mac
   # ou
   env\Scripts\activate  # Sur Windows
   ```

3. **Installez les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

   **Dépendances principales** :
   - **torch** (≥2.0.0) : Framework deep learning PyTorch
   - **pandas**, **numpy** : Manipulation de données
   - **scikit-learn** : Preprocessing et métriques
   - **plotly**, **matplotlib**, **seaborn** : Visualisation
   - **streamlit** : Interface web interactive
   - **yfinance** : Téléchargement des données boursières
   - **jupyter** : Environnement notebook

---

## Utilisation

### 1. **Notebook Jupyter** (Apprentissage et exploration)

Exécutez le notebook pédagogique complet :

```bash
source env/bin/activate  
jupyter notebook sp500_lstm_model.ipynb
```

Le notebook contient :
- Explication théorique des LSTM
- Préparation des données
- Entraînement du modèle
- Évaluation et prédictions
- Visualisations détaillées

### 2. **Dashboard Streamlit** (Interface interactive)

Lancez le dashboard interactif pour explorer les prédictions :

```bash
source env/bin/activate  # Activez l'environnement virtuel
streamlit run dashboard.py
```

**Fonctionnalités du dashboard** :
-  Visualisation des données historiques
-  Prédictions multi-horizons (1, 5, 10, 30 jours)
-  Intervalles de confiance et scénarios
-  Export en CSV/PDF
-  Monitoring en temps réel
-  Gamification (challenge de prédiction)

Le dashboard s'ouvrira automatiquement dans votre navigateur à : `http://localhost:8501`


---

## Structure du projet

```
projet/
├── sp500_lstm_model.ipynb              #  Notebook principal (pédagogique)
├── dashboard.py                        #  Interface Streamlit
├── src/                                #  Modules Python
│   ├── model.py                        # Classe LSTM et fonctions de sauvegarde
│   ├── data_loader.py                  # Chargement et préparation des données
│   ├── utils.py                        # Fonctions utilitaires
│   └── __init__.py
│
├── models/                             # Modèles entraînés
│   ├── lstm_model.pth                  # Poids du modèle LSTM
│   ├── best_model.pth                  # Meilleur modèle trouvé
│   └── config.json                     # Configuration du modèle
│
├── data/                               # Données
│   └── sp500_data.csv                  # Données historiques S&P500


```

---

## Scripts disponibles

| Script | Commande | Description |
|--------|----------|-------------|
| **Notebook** | `jupyter notebook sp500_lstm_model.ipynb` | Notebook complet et interactif |
| **Dashboard** | `streamlit run dashboard.py` | Interface web interactive |

