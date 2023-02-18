from dataclasses import dataclass
from neurosegmenter.plugins import register
from neurosegmenter.plugins import PluginType


class TestPlugin:
    name: str
    
    @staticmethod
    def initialize(name: str) -> None:
        pass
    

def initialize() -> None:
    register("test_plugin", PluginType.MODEL, TestPlugin)