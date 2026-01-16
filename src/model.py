"""
Module contenant la definition du modele LSTM.

Ce fichier est utilise a la fois par le notebook d'entrainement
et par le dashboard pour charger le modele.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Modele LSTM pour la prediction de series temporelles financieres.

    Architecture :
    - Couches LSTM empilees avec dropout pour la regularisation
    - Couche fully connected pour produire la prediction finale

    Attributes:
        hidden_size: Nombre de neurones dans chaque couche LSTM
        num_layers: Nombre de couches LSTM empilees
        lstm: Module LSTM de PyTorch
        fc: Couche lineaire de sortie
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=1):
        """
        Initialise le modele LSTM.

        Args:
            input_size: Nombre de features en entree (ex: 5 pour OHLCV)
            hidden_size: Nombre de neurones dans chaque couche LSTM
            num_layers: Nombre de couches LSTM empilees
            dropout: Taux de dropout entre les couches (defaut: 0.2)
            output_size: Nombre de valeurs a predire (defaut: 1)
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Couches LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Couche de sortie
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass du modele.

        Args:
            x: Tensor de shape [batch_size, sequence_length, input_size]

        Returns:
            Tensor de shape [batch_size] (predictions)
        """
        # Passage a travers le LSTM
        # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Prendre la sortie du dernier pas de temps
        last_output = lstm_out[:, -1, :]

        # Passage dans la couche fully connected
        output = self.fc(last_output)

        return output.squeeze()


def load_model(model_path, config, device='cpu'):
    """
    Charge un modele LSTM depuis un fichier.

    Args:
        model_path: Chemin vers le fichier .pth
        config: Dictionnaire de configuration avec les hyperparametres
        device: Device sur lequel charger le modele ('cpu' ou 'cuda')

    Returns:
        Modele LSTM charge en mode evaluation
    """
    # Extraire les parametres du modele depuis la config
    model_params = config.get('model_params', {})
    model_config = config.get('config', config)

    input_size = model_params.get('input_size', 5)
    hidden_size = model_params.get('hidden_size', model_config.get('hidden_size', 64))
    num_layers = model_params.get('num_layers', model_config.get('num_layers', 2))
    dropout = model_params.get('dropout', model_config.get('dropout', 0.2))

    # Creer le modele avec la bonne architecture
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1
    ).to(device)

    # Charger les poids
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Mode evaluation
    model.eval()

    return model
