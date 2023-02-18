"""Simple plugin loader."""
import importlib
from typing import Type, Any
from enum import Enum
from dataclasses import dataclass

from typing import Protocol
from typing import runtime_checkable

from tensorflow import keras
from keras import Model
class PluginType(Enum):
    MODEL = 1
    DATAGEN = 2
    LOSS = 3
    METRIC = 4
    OPTIMIZER = 5
    CALLBACK = 6

class PluginInterface:
    """Plugin interface, used in the plugin loader."""

    @staticmethod
    def initialize(*args, **kwargs):
        """Initialize the plugin."""
        raise NotImplementedError
    
@dataclass
class PluginParameter:
    """Base class for plugin parameters."""
    name: str # parameter name
    description: str # parameter description
    type: Type # parameter type, e.g. str, int, float, bool
    default: Any # default value
    path: str # path to the parameter in the configuration file, e.g. "test_plugin.name"
    

class Plugin(Protocol):
    """Protocol for plugins."""
    name: str # plugin name
    type: PluginType # plugin type, e.g. model, datagen, loss, metric, optimizer, callback
    description: str # plugin description
    parameters: list[PluginParameter] # plugin parameters

    
def import_plugin_module(plugin_name: str) -> PluginInterface:
    """Load a model plugin."""
    plugin = importlib.import_module(f"neurosegmenter.plugins.{plugin_name}")
    return plugin # type: ignore

def load_plugins(plugins: list[str]) -> None:
    """Load model plugins."""
    for plugin_name in plugins:
        plugin_interface = import_plugin_module(plugin_name)
        plugin_interface.initialize()
        

registered_plugins: dict[str, Plugin] = {}        
      
def register_plugin(plugin: Plugin) -> None:
    """Register a plugin."""
    registered_plugins[plugin.name] = plugin 
    
def unregister_plugin(plugin_name: str) -> None:
    """Unregister a plugin."""
    del registered_plugins[plugin_name]
    