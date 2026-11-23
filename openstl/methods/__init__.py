# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .predrnn import PredRNN
from .prednet import PredNet
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .simvp import SimVP
from .tau import TAU
from .mmvp import MMVP
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .wast import WaST

method_maps = {
    'convlstm': ConvLSTM,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'predrnn': PredRNN,
    'prednet': PredNet,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'simvp': SimVP,
    'tau': TAU,
    'mmvp': MMVP,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'wast': WaST
}

__all__ = [
    'method_maps', 'ConvLSTM', 'E3DLSTM', 'MAU', 'MIM',
    'PredNet', 'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'SimVP', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST'
]

from .pfg import PFG
method_maps['pfg'] = PFG
