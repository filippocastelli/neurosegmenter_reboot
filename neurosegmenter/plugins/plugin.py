"""Simple plugin loader."""
import importlib
from typing import Type
from enum import Enum

class PluginType(Enum):
    MODEL = 1
    DATAGEN = 2
    LOSS = 3
    METRIC = 4
    OPTIMIZER = 5
    CALLBACK = 6

class PluginInterface:
    """Interface for plugins."""

    @staticmethod
    def initialize(*args, **kwargs):
        """Initialize the plugin."""
        raise NotImplementedError
    
def import_plugin_module(plugin_name: str) -> PluginInterface:
    """Load a model plugin."""
    plugin = importlib.import_module(f"neurosegmenter.plugins.{plugin_name}")
    return plugin # type: ignore

def load_plugins(plugins: list[str]) -> None:
    """Load model plugins."""
    for plugin_name in plugins:
        plugin_interface = import_plugin_module(plugin_name)
        plugin_interface.initialize()
        

registered_plugins: dict[str, tuple[PluginType, PluginInterface]] = {}        
      
def register(plugin_name: str, plugin_type: PluginType, plugin_class: Type) -> None:
    """Register a plugin."""
    registered_plugins[plugin_name] = (plugin_type, plugin_class)
    
def unregister(plugin_name: str) -> None:
    """Unregister a plugin."""
    del registered_plugins[plugin_name]
    