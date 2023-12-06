import torch
import configargparse
import torch.nn as nn
from utils.encoder import TransformerEncoder
from utils.util import StatsCalculator
import torch.nn as nn

class CTCProjection(nn.Module):
    def __init__(self, odim: int, idim: int):
        """Calculate CTC loss for the output of the encoder
        :param odim: output dimension, i.e. vocabulary size including the blank label
        :param idim: input dimension of ctc linear layer
        """
        super().__init__()
        self.projection = nn.Linear(idim, odim)

class ASRModel(torch.nn.Module):
    def __init__(self, params: configargparse.Namespace):
        """E2E ASR model implementation.

        Args:
            params: The training options
        """
        super().__init__()

        self.ignore_id = params.text_pad

        self.encoder = TransformerEncoder(
            input_size=params.idim,
            output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.eblocks,
            dropout_rate=params.edropout,
            positional_dropout_rate=params.edropout,
            attention_dropout_rate=params.edropout,
        )
        self.ctc = CTCProjection(odim=params.odim, idim=params.hidden_dim)  # just a projection layer, no loss
        self.stat_calculator = StatsCalculator(params)