"""
Slim wrapper around ccxt's Precise (string math)
To have imports from trading - and support float initializers
"""
from ccxt import Precise


class FtPrecise(Precise):
    def __init__(self, number, decimals=None):
        if not isinstance(number, str):
            number = str(number)
        super().__init__(number, decimals)
