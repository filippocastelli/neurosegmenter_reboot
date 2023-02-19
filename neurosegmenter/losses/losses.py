from neurosegmenter.losses import Loss
from neurosegmenter.plugins import Plugin, PluginType, PluginParameter
from neurosegmenter.plugins import register_plugin
from tensorflow import keras
from keras import metrics
from keras import losses
import numpy as np
import tensorflow as tf

plugins = []

class BinaryCrossEntropyLoss(Plugin, Loss):
    name: str = "Binary Cross Entropy Loss"
    description: str = "Binary Cross Entropy Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.binary_cross_entropy_loss"
    parameters: list[PluginParameter] = [
        PluginParameter(name ="from_logits",
                        description = "Whether y_pred is expected to be a logits tensor. By default, we assume that y_pred encodes a probability distribution.",
                        type = bool,
                        default = True,
                        path = "losses.binary_cross_entropy_loss.from_logits"),
        PluginParameter(name ="label_smoothing",
                        description="Float in [0, 1]. If > 0 then smooth the labels.",
                        default=0.0,
                        type=float,
                        path= "losses.binary_cross_entropy_loss.label_smoothing"),
        PluginParameter(name ="axis",
                        description="The dimension along which the entropy is computed.",
                        type=int,
                        default=-1,
                        path="losses.binary_cross_entropy_loss.axis")
    ]
    
    from_logits: bool
    label_smoothing: float
    axis: int

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metrics.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing, axis=self.axis)

plugins.append(BinaryCrossEntropyLoss)

class MeanSquaredErrorLoss(Plugin, Loss):
    name: str = "Mean Squared Error Loss"
    description: str = "Mean Squared Error Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.mean_squared_error_loss"
    parameters: list[PluginParameter] = []
    
    def __init__(self):
        super().__init__()
        self.mse = losses.MeanSquaredError()
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.mse(y_true, y_pred)
    
plugins.append(MeanSquaredErrorLoss)


class MeanAbsoluteErrorLoss(Plugin, Loss):
    name: str = "Mean Absolute Error Loss"
    description: str = "Mean Absolute Error Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.mean_absolute_error_loss"
    parameters: list[PluginParameter] = []
    
    def __init__(self):
        super().__init__()
        self.mae = losses.MeanAbsoluteError()
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.mae(y_true, y_pred)    
    
plugins.append(MeanAbsoluteErrorLoss)


class MeanAbsolutePercentageErrorLoss(Plugin, Loss):
    name: str = "Mean Absolute Percentage Error Loss"
    description: str = "Mean Absolute Percentage Error Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.mean_absolute_percentage_error_loss"
    parameters: list[PluginParameter] = []
    
    def __init__(self):
        super().__init__()
        self.mape = losses.MeanAbsolutePercentageError()
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.mape(y_true, y_pred)
    
plugins.append(MeanAbsolutePercentageErrorLoss)

class MeanSquaredLogarithmicErrorLoss(Plugin, Loss):
    name: str = "Mean Squared Logarithmic Error Loss"
    description: str = "Mean Squared Logarithmic Error Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.mean_squared_logarithmic_error_loss"
    parameters: list[PluginParameter] = []
    
    def __init__(self):
        super().__init__()
        self.msle = losses.MeanSquaredLogarithmicError()
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.msle(y_true, y_pred)
    
plugins.append(MeanSquaredLogarithmicErrorLoss)


class CosineSimilarityLoss(Plugin, Loss):
    name: str = "Cosine Similarity Loss"
    description: str = "Cosine Similarity Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.cosine_similarity_loss"
    parameters: list[PluginParameter] = [
        PluginParameter(name ="axis",
                        description="Axis along which to determine similarity.",
                        type=int,
                        default=-1,
                        path="losses.cosine_similarity_loss.axis")
    ]
    
    axis: int

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return losses.cosine_similarity(y_true, y_pred, axis=self.axis)
    
    
plugins.append(CosineSimilarityLoss)


class HingeLoss(Plugin, Loss):
    name: str = "Hinge Loss"
    description: str = "Hinge Loss"
    type: PluginType = PluginType.LOSS
    path: str = "losses.hinge_loss"
    parameters: list[PluginParameter] = []
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return losses.hinge(y_true, y_pred)
    
    
plugins.append(HingeLoss)


for plugin in plugins:
    register_plugin(plugin)