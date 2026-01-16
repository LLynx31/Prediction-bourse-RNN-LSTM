"""
Dashboard Streamlit pour la visualisation des predictions S&P500.

Ce dashboard permet de :
- Visualiser les donnees historiques du S&P500
- Afficher les predictions du modele LSTM
- Comparer les predictions aux valeurs reelles
- Voir les metriques de performance

Lancement : streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import json
import joblib
import os
from datetime import datetime, timedelta

# Import des modules locaux
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import LSTMModel, load_model
from src.data_loader import (
    download_data,
    load_data_from_csv,
    prepare_data,
    create_sequences,
    inverse_transform_close,
    get_last_sequence
)
from src.utils import (
    calculate_metrics,
    format_currency,
    format_percentage,
    calculate_change,
    get_trend_indicator
)

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================

st.set_page_config(
    page_title="S&P500 LSTM Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# FONCTIONS DE CHARGEMENT
# ============================================

@st.cache_resource
def load_model_and_config():
    """Charge le modele et la configuration (mis en cache)."""
    config_path = 'models/config.json'
    model_path = 'models/lstm_model.pth'
    scaler_path = 'models/scaler.pkl'

    # Verifier que les fichiers existent
    if not all(os.path.exists(p) for p in [config_path, model_path, scaler_path]):
        return None, None, None

    # Charger la configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Charger le modele
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, config, device)

    # Charger le scaler
    scaler = joblib.load(scaler_path)

    return model, config, scaler


@st.cache_data(ttl=3600)  # Cache d'une heure
def load_stock_data():
    """Charge les donnees du S&P500 (mis en cache)."""
    data_path = 'data/sp500_data.csv'

    if os.path.exists(data_path):
        df = load_data_from_csv(data_path)
    else:
        df = download_data('^GSPC', years=5)

    return df


def make_predictions(model, df, scaler, config, device='cpu'):
    """Genere les predictions sur les donnees."""
    sequence_length = config['config']['sequence_length']

    # Preparer les donnees
    data_scaled, _ = prepare_data(df, scaler=scaler, fit_scaler=False)

    # Creer les sequences
    X, y = create_sequences(data_scaled, sequence_length)

    # Convertir en tenseurs
    X_tensor = torch.FloatTensor(X).to(device)

    # Predictions
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()

    # Denormaliser
    predictions = inverse_transform_close(predictions_scaled, scaler)
    actual = inverse_transform_close(y, scaler)

    # Creer un DataFrame avec les dates
    dates = df.index[sequence_length:]
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predictions
    })
    results_df.set_index('Date', inplace=True)

    return results_df


# ============================================
# COMPOSANTS DE L'INTERFACE
# ============================================

def render_sidebar(config):
    """Affiche la barre laterale."""
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.radio(
        "Choisir une section",
        ["Donnees Historiques", "Predictions", "A propos du Modele"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Informations")

    if config:
        model_config = config.get('config', config)
        model_params = config.get('model_params', {})
        st.sidebar.markdown(f"""
        **Modele LSTM**
        - Couches : {model_params.get('num_layers', model_config.get('num_layers', 'N/A'))}
        - Neurones : {model_params.get('hidden_size', model_config.get('hidden_size', 'N/A'))}
        - Sequence : {model_config.get('sequence_length', 'N/A')} jours

        **Entrainement**
        - Date : {config.get('training_date', 'N/A')}
        - Epochs : {model_config.get('epochs', 'N/A')}
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Projet pedagogique<br>Deep Learning - LSTM</small>
    </div>
    """, unsafe_allow_html=True)

    return page


def render_historical_data(df):
    """Affiche les donnees historiques."""
    st.markdown("## Donnees Historiques du S&P500")

    # Selection de la periode
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox(
            "Periode a afficher",
            ["1 mois", "3 mois", "6 mois", "1 an", "2 ans", "Tout"]
        )

    # Filtrer les donnees selon la periode
    periods_days = {
        "1 mois": 30,
        "3 mois": 90,
        "6 mois": 180,
        "1 an": 365,
        "2 ans": 730,
        "Tout": len(df)
    }
    days = periods_days[period]
    df_filtered = df.tail(days)

    # Graphique Candlestick
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Prix S&P500', 'Volume')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_filtered.index,
            open=df_filtered['Open'],
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green'
              for _, row in df_filtered.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df_filtered.index,
            y=df_filtered['Volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistiques
    st.markdown("### Statistiques")
    col1, col2, col3, col4 = st.columns(4)

    last_close = df_filtered['Close'].iloc[-1]
    first_close = df_filtered['Close'].iloc[0]
    change, change_pct = calculate_change(last_close, first_close)

    with col1:
        st.metric(
            "Dernier Prix",
            format_currency(last_close),
            f"{format_percentage(change_pct)}"
        )

    with col2:
        st.metric(
            "Plus Haut",
            format_currency(df_filtered['High'].max())
        )

    with col3:
        st.metric(
            "Plus Bas",
            format_currency(df_filtered['Low'].min())
        )

    with col4:
        st.metric(
            "Volume Moyen",
            f"{df_filtered['Volume'].mean()/1e9:.2f}B"
        )


def render_predictions(df, model, scaler, config):
    """Affiche les predictions."""
    st.markdown("## Predictions du Modele LSTM")

    # Avertissement
    st.markdown("""
    <div class="warning-box">
        <strong>Avertissement</strong> : Ces predictions sont a but pedagogique uniquement.
        Ne jamais utiliser ce modele pour des decisions d'investissement reelles.
    </div>
    """, unsafe_allow_html=True)

    # Generer les predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = make_predictions(model, df, scaler, config, device)

    # Split train/test
    train_size = int(len(results) * 0.8)
    results_test = results.iloc[train_size:]

    # Prediction du prochain jour
    st.markdown("### Prediction du Prochain Jour")

    model_config = config.get('config', config)
    sequence_length = model_config.get('sequence_length', 60)

    last_sequence = get_last_sequence(df, scaler, sequence_length)
    last_sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        next_pred_scaled = model(last_sequence_tensor).item()

    next_prediction = inverse_transform_close(
        np.array([next_pred_scaled]),
        scaler
    )[0]

    last_price = df['Close'].iloc[-1]
    pred_change, pred_change_pct = calculate_change(next_prediction, last_price)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Prix Actuel</h3>
            <h2>{format_currency(last_price)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Prediction Demain</h3>
            <h2>{format_currency(next_prediction)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        indicator = get_trend_indicator(pred_change)
        color = 'green' if pred_change >= 0 else 'red'
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Variation Prevue</h3>
            <h2 style="color: {color}">{indicator} {format_percentage(pred_change_pct)}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Graphique comparatif
    st.markdown("### Comparaison Reel vs Predit (Jeu de Test)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_test.index,
        y=results_test['Actual'],
        mode='lines',
        name='Valeur Reelle',
        line=dict(color='#2E86AB', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=results_test.index,
        y=results_test['Predicted'],
        mode='lines',
        name='Prediction LSTM',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))

    fig.update_layout(
        height=500,
        xaxis_title='Date',
        yaxis_title='Prix ($)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metriques de performance
    st.markdown("### Metriques de Performance")

    metrics = calculate_metrics(results_test['Actual'].values, results_test['Predicted'].values)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
        st.caption("Root Mean Squared Error")

    with col2:
        st.metric("MAE", f"${metrics['MAE']:.2f}")
        st.caption("Mean Absolute Error")

    with col3:
        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        st.caption("Mean Absolute Percentage Error")

    # Scatter plot
    st.markdown("### Precision des Predictions")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=results_test['Actual'],
        y=results_test['Predicted'],
        mode='markers',
        marker=dict(
            color='#2E86AB',
            size=8,
            opacity=0.6
        ),
        name='Predictions'
    ))

    # Ligne parfaite
    min_val = min(results_test['Actual'].min(), results_test['Predicted'].min())
    max_val = max(results_test['Actual'].max(), results_test['Predicted'].max())

    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Prediction Parfaite'
    ))

    fig_scatter.update_layout(
        height=500,
        xaxis_title='Valeur Reelle ($)',
        yaxis_title='Valeur Predite ($)',
        template='plotly_white'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


def render_about(config):
    """Affiche la section A propos."""
    st.markdown("## A propos du Modele")

    st.markdown("""
    ### Qu'est-ce qu'un LSTM ?

    Le **LSTM** (Long Short-Term Memory) est un type de reseau de neurones recurrent
    concu pour traiter des sequences temporelles. Contrairement aux reseaux classiques,
    le LSTM peut "memoriser" des informations sur de longues periodes grace a son
    systeme de portes (gates).

    ```
    +-------------------------------------------------------------+
    |                    Cellule LSTM                             |
    +-------------------------------------------------------------+
    |                                                             |
    |   +----------+   +----------+   +----------+               |
    |   |  Forget  |   |  Input   |   |  Output  |               |
    |   |   Gate   |   |   Gate   |   |   Gate   |               |
    |   +----+-----+   +----+-----+   +----+-----+               |
    |        |              |              |                      |
    |        v              v              v                      |
    |   +---------------------------------------------+          |
    |   |          Cell State (memoire)               |          |
    |   +---------------------------------------------+          |
    |                                                             |
    |   - Forget Gate : Decide quoi oublier                      |
    |   - Input Gate  : Decide quoi memoriser                    |
    |   - Output Gate : Decide quoi utiliser                     |
    |                                                             |
    +-------------------------------------------------------------+
    ```
    """)

    st.markdown("### Architecture du Modele")

    if config:
        model_config = config.get('config', config)
        model_params = config.get('model_params', {})
        data_info = config.get('data_info', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Hyperparametres")
            st.markdown(f"""
            | Parametre | Valeur |
            |-----------|--------|
            | Input Size | {model_params.get('input_size', 5)} features |
            | Hidden Size | {model_params.get('hidden_size', model_config.get('hidden_size', 64))} neurones |
            | Num Layers | {model_params.get('num_layers', model_config.get('num_layers', 2))} couches |
            | Dropout | {model_params.get('dropout', model_config.get('dropout', 0.2))} |
            | Sequence Length | {model_config.get('sequence_length', 60)} jours |
            """)

        with col2:
            st.markdown("#### Features utilisees")
            features = data_info.get('features', ['Open', 'High', 'Low', 'Close', 'Volume'])
            for i, feat in enumerate(features, 1):
                st.markdown(f"{i}. **{feat}**")

    st.markdown("### Limitations")

    st.markdown("""
    Ce modele est **purement pedagogique**. Voici pourquoi il ne peut pas predire
    de maniere fiable les marches financiers :

    1. **Efficience des marches** : Les prix integrent deja toute l'information disponible
    2. **Facteurs imprevisibles** : Guerres, pandemies, decisions politiques...
    3. **Bruit vs Signal** : La majorite des mouvements quotidiens est du bruit aleatoire
    4. **Overfitting temporel** : Le modele peut memoriser le passe sans generaliser

    **Ne jamais utiliser ce modele pour des decisions d'investissement reelles !**
    """)


# ============================================
# APPLICATION PRINCIPALE
# ============================================

def main():
    """Point d'entree de l'application."""
    # En-tete
    st.markdown('<h1 class="main-header">S&P500 LSTM Predictor</h1>', unsafe_allow_html=True)

    # Chargement du modele
    model, config, scaler = load_model_and_config()

    if model is None:
        st.error("""
        **Modele non trouve !**

        Veuillez d'abord executer le notebook `sp500_lstm_model.ipynb` pour entrainer
        le modele et sauvegarder les fichiers necessaires.

        Fichiers requis :
        - `models/lstm_model.pth`
        - `models/scaler.pkl`
        - `models/config.json`
        """)
        return

    # Chargement des donnees
    try:
        df = load_stock_data()
    except Exception as e:
        st.error(f"Erreur lors du chargement des donnees : {e}")
        return

    # Sidebar et navigation
    page = render_sidebar(config)

    # Affichage de la page selectionnee
    if page == "Donnees Historiques":
        render_historical_data(df)
    elif page == "Predictions":
        render_predictions(df, model, scaler, config)
    elif page == "A propos du Modele":
        render_about(config)


if __name__ == "__main__":
    main()
