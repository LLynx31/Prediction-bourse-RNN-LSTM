"""
Module pour le chargement et le pretraitement des donnees.

Ce fichier contient les fonctions pour :
- Telecharger les donnees depuis Yahoo Finance
- Preparer les donnees pour le modele LSTM
- Creer les sequences temporelles
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib


def download_data(ticker='^GSPC', years=5):
    """
    Telecharge les donnees historiques depuis Yahoo Finance.

    Args:
        ticker: Symbole boursier (defaut: '^GSPC' pour S&P500)
        years: Nombre d'annees d'historique

    Returns:
        DataFrame avec les colonnes Open, High, Low, Close, Volume
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    # Gestion du multi-index si present
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)

    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[columns_to_keep]

    return data


def load_data_from_csv(filepath):
    """
    Charge les donnees depuis un fichier CSV.

    Args:
        filepath: Chemin vers le fichier CSV

    Returns:
        DataFrame avec les donnees
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def prepare_data(df, scaler=None, fit_scaler=True):
    """
    Prepare les donnees pour le modele LSTM.

    Args:
        df: DataFrame avec les colonnes OHLCV
        scaler: MinMaxScaler existant (optionnel)
        fit_scaler: Si True, fit le scaler sur les donnees

    Returns:
        Tuple (donnees normalisees, scaler)
    """
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[feature_columns].values

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if fit_scaler:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    return data_scaled, scaler


def create_sequences(data, sequence_length=60):
    """
    Cree des sequences pour l'entrainement du LSTM.

    Args:
        data: Donnees normalisees (array 2D)
        sequence_length: Longueur de chaque sequence

    Returns:
        Tuple (X, y) ou X sont les sequences et y les cibles
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 3])  # Index 3 = Close

    return np.array(X), np.array(y)


def inverse_transform_close(scaled_values, scaler, n_features=5, close_idx=3):
    """
    Denormalise les valeurs du prix Close.

    Args:
        scaled_values: Valeurs normalisees a denormaliser
        scaler: MinMaxScaler utilise pour la normalisation
        n_features: Nombre total de features (defaut: 5)
        close_idx: Index de la colonne Close (defaut: 3)

    Returns:
        Array des valeurs denormalisees
    """
    scaled_values = np.array(scaled_values).flatten()
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, close_idx] = scaled_values
    dummy_inverse = scaler.inverse_transform(dummy)
    return dummy_inverse[:, close_idx]


def get_last_sequence(df, scaler, sequence_length=60):
    """
    Recupere la derniere sequence pour faire une prediction.

    Args:
        df: DataFrame avec les donnees
        scaler: MinMaxScaler pour normaliser
        sequence_length: Longueur de la sequence

    Returns:
        Array normalise de la derniere sequence
    """
    data_scaled, _ = prepare_data(df, scaler=scaler, fit_scaler=False)
    last_sequence = data_scaled[-sequence_length:]
    return last_sequence
