"""
Module utilitaire contenant des fonctions diverses.

Ce fichier contient :
- Fonctions de calcul de metriques
- Fonctions de formatage
- Utilitaires divers
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(actual, predicted):
    """
    Calcule les metriques d'evaluation du modele.

    Args:
        actual: Valeurs reelles
        predicted: Valeurs predites

    Returns:
        Dictionnaire avec RMSE, MAE et MAPE
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # Eviter la division par zero
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def format_currency(value):
    """
    Formate une valeur en dollars.

    Args:
        value: Valeur numerique

    Returns:
        String formatee (ex: "$1,234.56")
    """
    return f"${value:,.2f}"


def format_percentage(value, decimals=2):
    """
    Formate une valeur en pourcentage.

    Args:
        value: Valeur numerique (ex: 5.5 pour 5.5%)
        decimals: Nombre de decimales

    Returns:
        String formatee (ex: "+5.50%" ou "-3.20%")
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def calculate_change(current, previous):
    """
    Calcule le changement entre deux valeurs.

    Args:
        current: Valeur actuelle
        previous: Valeur precedente

    Returns:
        Tuple (changement absolu, changement en %)
    """
    change = current - previous
    change_pct = (change / previous) * 100 if previous != 0 else 0
    return change, change_pct


def get_trend_color(value):
    """
    Retourne une couleur basee sur le signe de la valeur.

    Args:
        value: Valeur numerique

    Returns:
        String couleur ('green', 'red', ou 'gray')
    """
    if value > 0:
        return 'green'
    elif value < 0:
        return 'red'
    return 'gray'


def get_trend_indicator(value):
    """
    Retourne un indicateur de tendance base sur le signe de la valeur.

    Args:
        value: Valeur numerique

    Returns:
        String indicateur (fleche haut, bas ou droite)
    """
    if value > 0:
        return '^'  # Fleche vers le haut
    elif value < 0:
        return 'v'  # Fleche vers le bas
    return '-'  # Stable
