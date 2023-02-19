from typing import Protocol
from tensorflow import keras
from keras import optimizers as keras_optimizers

class Optimizer(Protocol):
    def get_optimizer(self, *args, **kwargs) -> keras_optimizers.Optimizer:
        ...