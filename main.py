from neurosegmenter.plugins import load_plugins, registered_plugins


load_plugins(["test_plugin"])
print(registered_plugins)


