__author__ = "R Devon Hjelm"
__copyright__ = "Copyright (c) Microsoft Corporation and Mila: The Quebec " \
                "AI Company"
__license__ = "MIT"

from .inception_generator import InceptionClusterGenerator
from .linear_autoencoder import LinearAEGenerator
from .generator import Generator

__all__ = ('InceptionClusterGenerator', 'LinearAEGenerator', 'Generator')
