import tensorflow as tf
from neurosegmenter.models import TrainableModel
from neurosegmenter.plugins import Plugin
from neurosegmenter.plugins import PluginType, PluginParameter

from tensorflow import keras
from keras import Model
from keras import layers

from neurosegmenter.plugins import register_plugin

class SimpleModelFromKerasModelClass(TrainableModel, Plugin):
    """Example model plugin that uses a keras model class
    
    note: the actual keras.Model class is defined inside the plugin class
    to allow for the plugin parameters to be passed to the model class
    the model class is instantiated inside the get_model method and needs
    the plugin instance as an argument.
    
    """
    name: str = "simple_model_from_keras_model"
    description: str = "Simple test model"
    type: PluginType = PluginType.MODEL
    parameters: list[PluginParameter] = [
        PluginParameter(name="activation",
                        description="Activation function",
                        type=str,
                        default="relu",
                        path="simple_model_from_keras_model.activation")
    ]
    
    activation: str
    
    def __init__(self) -> None:
        for param in self.parameters:
            setattr(self, param.name, param.default)
    
    def createKerasModel(self):
        return SimpleModelFromKerasModelClass.SimpleKerasModel(self)
    
    class SimpleKerasModel(tf.keras.Model):
        """good ol' keras model class, but with a parent attribute in the constructor"""
        def __init__(self, parent): 
            self.parent = parent 
            super().__init__()
            self.conv1 = layers.Conv2D(32, 3, activation=self.parent.activation)
            self.flatten = layers.Flatten()
            self.d1 = layers.Dense(128, activation=self.parent.activation)
            self.d2 = layers.Dense(10, activation='softmax')
        
        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)
        
    def get_model(self) -> Model:
        return self.createKerasModel()

class SimpleModelFromKerasFunctional(TrainableModel, Plugin):
    """Example model plugin that uses keras's functional API"""
    
    name : str = "simple_model_from_keras_functional"
    description: str = "Simple model from keras functional API"
    parameters: list[PluginParameter] = [
        PluginParameter(name="activation",
                        description="Activation function",
                        type=str,
                        default="relu",
                        path="simple_model_from_keras_model.activation")
    ]
    type: PluginType = PluginType.MODEL
    
    activation: str
    
    def __init__(self) -> None:
        for param in self.parameters:
            setattr(self, param.name, param.default)
            
    def get_model(self) -> Model:
        """just a simple model with a single conv layer and a dense layer"""
        inputs = layers.Input(shape=(None, None, 1))
        processed = layers.RandomCrop(width=32, height=32)(inputs)
        conv = layers.Conv2D(filters=2, kernel_size=3, activation=self.activation)(processed)
        pooling = layers.GlobalAveragePooling2D()(conv)
        feature = layers.Dense(10, activation=self.activation)(pooling)
        return Model(inputs, feature)
    

# register the plugins
register_plugin(SimpleModelFromKerasModelClass)
register_plugin(SimpleModelFromKerasFunctional)