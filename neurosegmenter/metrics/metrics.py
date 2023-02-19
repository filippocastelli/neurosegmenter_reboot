from neurosegmenter.metrics import Metric
from neurosegmenter.plugins import Plugin, PluginType, PluginParameter
from neurosegmenter.plugins import register_plugin
from tensorflow import keras
from keras import backend as K
from keras import metrics
import numpy as np

plugins = []

class BinaryAccuracy(Plugin, Metric):
    name: str = "Binary Accuracy"
    description: str = "Binary Accuracy"
    type: PluginType = PluginType.METRIC
    path: str = "metrics.binary_accuracy"
    parameters: list[PluginParameter] = [
        PluginParameter(name ="threshold",
                        description = "Float representing the threshold for deciding whether the prediction value is above or below the threshold.",
                        type = float,
                        default = 0.5,
                        path = "metrics.binary_accuracy.threshold"),
    ]
    
    threshold: float
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metrics.binary_accuracy(y_true, y_pred, threshold=self.threshold)

plugins.append(BinaryAccuracy)


class CategoricalAccuracy(Plugin, Metric):
    name: str = "Categorical Accuracy"
    description: str = "Categorical Accuracy"
    type: PluginType = PluginType.METRIC
    path: str = "metrics.categorical_accuracy"
    parameters: list[PluginParameter] = []
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metrics.categorical_accuracy(y_true, y_pred)

plugins.append(CategoricalAccuracy)
    
class JaccardIndex(Plugin, Metric):
    name: str = "Jaccard Index"
    description: str = "Jaccard Index"
    type: PluginType = PluginType.METRIC
    path: str = "metrics.jaccard_index"
    parameters: list[PluginParameter] = []
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        
        dot_product = K.dot(y_true_flat, y_pred_flat)
        intersection = K.sum(dot_product)
        union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection
        
        iou = intersection / (union + K.epsilon()) # add epsilon to avoid division by zero
        return iou

plugins.append(JaccardIndex)
    
class DiceCoefficient(Plugin, Metric):
    name: str = "Dice Coefficient"
    description: str = "Dice Coefficient"
    type: PluginType = PluginType.METRIC
    path: str = "metrics.dice_coefficient"
    parameters: list[PluginParameter] = []
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        
        dot_product = K.dot(y_true_flat, y_pred_flat)
        intersection = K.sum(dot_product)
        union = K.sum(y_true_flat) + K.sum(y_pred_flat)
        
        dice = (2. * intersection) / (union + K.epsilon()) # add epsilon to avoid division by zero
        return dice

plugins.append(DiceCoefficient)
    
    
class BinaryCrossEntropy(Plugin, Metric):
    name: str = "Binary Cross Entropy"
    description: str = "Binary Cross Entropy"
    type: PluginType = PluginType.METRIC
    path: str = "metrics.binary_cross_entropy"
    parameters: list[PluginParameter] = [
        PluginParameter(name ="from_logits",
                        description = "Whether y_pred is expected to be a logits tensor. By default, we assume that y_pred encodes a probability distribution.",
                        type = bool,
                        default = True,
                        path = "metrics.binary_cross_entropy.from_logits"),
        PluginParameter(name ="label_smoothing",
                        description="Float in [0, 1]. If > 0 then smooth the labels.",
                        default=0.0,
                        type=float,
                        path= "metrics.binary_cross_entropy.label_smoothing"),
        PluginParameter(name ="axis",
                        description="The dimension along which the entropy is computed.",
                        type=int,
                        default=-1,
                        path="metrics.binary_cross_entropy.axis")
    ]
    
    from_logits: bool
    label_smoothing: float
    axis: int

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metrics.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing, axis=self.axis)

plugins.append(BinaryCrossEntropy)


for plugin in plugins:
    register_plugin(plugin)