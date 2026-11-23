"""The model plugin of Inkling-2 looped checkpoints: the layout beside this file (layout.json) composed by the models core
(models/core/model-template, the plugin this one extends) from the model techniques it names."""
from pathlib import Path

from .core.template.compose import model_plugin

resolve, memory, create = model_plugin(Path(__file__).resolve().parent)
