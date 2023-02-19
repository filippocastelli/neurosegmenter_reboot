from neurosegmenter.plugins import register_plugin
from neurosegmenter.plugins import PluginType
from neurosegmenter.plugins import PluginParameter
from neurosegmenter.plugins import Plugin

from neurosegmenter.models import TrainableModel
from neurosegmenter.metrics import Metric

from tensorflow import keras
from keras import Model

import numpy as np
class TestPluginModel(TrainableModel, Plugin):
    name: str = "test_plugin_model"
    description: str = "Test plugin"
    type: PluginType = PluginType.MODEL
    parameters: list[PluginParameter] = [
        PluginParameter(name="test_str",
                        description="example string parameter",
                        type=str,
                        default="test_plugin",
                        path="test_plugin.name"),
        
        PluginParameter(name="test_int",
                        description="example integer parameter",
                        type=int,
                        default=1,
                        path="test_plugin.test_int"),
        ]
    
    def get_model(self) -> Model:
        return Model()

class TestPluginMetric(Plugin, Metric):
    name: str = "test_plugin_metric"
    description: str = "Test plugin"
    type: PluginType = PluginType.METRIC
    parameters: list[PluginParameter] = [
        PluginParameter(name="test_str",
                        description="example string parameter",
                        type=str,
                        default="test_plugin",
                        path="test_plugin.name"),
        
        PluginParameter(name="test_int",
                        description="example integer parameter",
                        type=int,
                        default=1,
                        path="test_plugin.test_int"),
        ]
            
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.array([1, 2, 3])
            
            
def initialize() -> None:
    register_plugin(TestPluginModel) 
    register_plugin(TestPluginMetric)