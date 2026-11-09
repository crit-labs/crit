"""crit plugins: every engine is Core plus plugins that fill its slots, put together when it loads (docs/PLUGINS.md)."""
from .assemble import Assembly, Registry, assemble, check, normalize
from .device import DEVICE_INTERFACES, Allowed, StepBatch
from .manifest import DEVICES, POLICY_SLOTS, REQUIRED_SLOTS, SLOTS, PluginError, load_manifest
from .slots import IMAGES, Image, model_problems

__all__ = ["Allowed", "Assembly", "DEVICES", "DEVICE_INTERFACES", "IMAGES", "Image", "POLICY_SLOTS", "PluginError",
           "REQUIRED_SLOTS", "Registry", "SLOTS", "StepBatch", "assemble", "check", "load_manifest", "model_problems",
           "normalize"]
