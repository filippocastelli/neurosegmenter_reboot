from tensorflow import keras
from keras.models import Model
from typing import Protocol
from neurosegmenter.plugins import registered_plugins

class TrainableModel(Protocol):
    def get_model(self) -> Model:
        ...
    