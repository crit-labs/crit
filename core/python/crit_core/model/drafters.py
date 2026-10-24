"""Published companion drafters that read the target's hidden states, as one description whatever their config format.

Three drafter kinds share one runtime contract:

- ``eagle3``: an autoregressive drafter (usually one decoder layer) whose input at position p is the target's hidden
  states at p from a few layers, concatenated and projected, with the embedding of the token at p + 1.
- ``block``: a block drafter (DFlash, DSpark) that drafts a whole block in one non-causal pass over
  [anchor, mask x (block_size - 1)], with the same target hidden states injected as context.
- ``mtp-head``: a head ``crit models drafter`` distilled from the target (``crit_drafter`` in its config.json): the
  EAGLE alignment over one tap, the target's final hidden state before its final norm, through the target's own head.

Taps are residual-stream indices: tap k is the hidden state after k target decoder layers (index k of transformers'
``hidden_states``: 0 is the embeddings). Formats spell them differently, and this table is the one place that knows:
EAGLE-3 ``eagle_aux_hidden_state_layer_ids`` and the speculators ``aux_hidden_state_layer_ids`` are taps already,
while native DFlash/DSpark ``target_layer_ids`` name the decoder layer whose output is read, so tap = id + 1.

Formats read: speculators (``speculators_model_type`` eagle3, peagle, dflash, dflash2, dspark, with
``transformer_layer_config`` and ``speculators_config``), native EAGLE-3 (an ``Eagle3`` architecture, ``eagle_config``
or top-level keys; default taps (2, L // 2, L - 3) for an L-layer target), native DFlash (``dflash_config``) and native
DSpark (a ``DSpark`` architecture) and crit's own drafters (``crit_drafter``). Anything else is not a hidden-state
drafter (read_drafter returns None).
"""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .. import jsonfile
from .config import CONFIG_NAME, MAX_CONFIG_BYTES, ConfigError

EAGLE3, BLOCK, MTP_HEAD = "eagle3", "block", "mtp-head"
_SPECULATORS = {"eagle3": EAGLE3, "peagle": EAGLE3, "dflash": BLOCK, "dflash2": BLOCK, "dspark": BLOCK}


@dataclass(frozen=True)
class Drafter:
    kind: str                       # EAGLE3, BLOCK or MTP_HEAD
    taps: tuple                     # residual-stream indices into the target (after k decoder layers)
    hidden_size: int | None         # the drafter's own width
    target_hidden_size: int | None  # the target's width the taps carry (None: the drafter's)
    draft_vocab_size: int | None    # a reduced draft vocabulary mapped back by d2t; None: the target's
    layers: int | None              # drafter decoder layers
    block_size: int | None          # block drafters: tokens per pass, anchor included
    mask_token_id: int | None       # block drafters (and parallel EAGLE): the mask placeholder
    sample_from_anchor: bool        # block drafters: the anchor position predicts too (block_size drafts, not - 1)
    causal: bool                    # block drafters: causal attention inside the block (most are not)
    confidence_head: bool           # a head that predicts each draft's acceptance
    speculative_tokens: int | None  # the publisher's recommended draft width
    verifier: str | None            # the target the drafter was trained for
    source: str                     # which format described it

    @property
    def drafts_per_pass(self):
        """Draft tokens one drafter pass yields: 1 for EAGLE-3 and MTP heads, block_size - 1 (or block_size) for block
        drafters."""
        if self.kind != BLOCK:
            return 1
        return self.block_size if self.sample_from_anchor else self.block_size - 1


def _int(value, source, key, minimum=0):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{source}: key '{key}': expected an integer >= {minimum}, not {value!r}")
    return value


def _optional_int(config, key, source, minimum=0):
    value = config.get(key)
    return None if value is None else _int(value, source, key, minimum)


def _ids(value, source, key):
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{source}: key '{key}': expected a nonempty list of layer indices")
    return tuple(_int(item, source, key) for item in value)


def _bool(config, key, source, default):
    value = config.get(key, default)
    if type(value) is not bool:
        raise ConfigError(f"{source}: key '{key}': expected a boolean, not {value!r}")
    return value


def _architectures(config):
    value = config.get("architectures") or []
    return [a for a in value if isinstance(a, str)]


def _checked(drafter, target_layers, source):
    if drafter.kind == BLOCK and (drafter.block_size is None or drafter.block_size < 2 or drafter.mask_token_id is None):
        raise ConfigError(f"{source}: a block drafter needs block_size >= 2 and mask_token_id")
    if target_layers is not None and any(t > target_layers for t in drafter.taps):
        raise ConfigError(f"{source}: taps {list(drafter.taps)} reach past the target's {target_layers} layers")
    return drafter


def _default_taps(target_layers, source, key):
    """EAGLE-3's default taps for an L-layer target, (2, L // 2, L - 3), as its reference implementations choose."""
    if target_layers is None:
        raise ConfigError(f"{source}: EAGLE-3 without {key} needs the target's layer count for its default taps")
    return (2, target_layers // 2, target_layers - 3)


def _speculators(config, source, target_layers):
    kind = _SPECULATORS.get(config["speculators_model_type"])
    if kind is None:
        return None
    layer = config.get("transformer_layer_config") or {}
    if not isinstance(layer, Mapping):
        raise ConfigError(f"{source}: key 'transformer_layer_config': expected an object")
    spec = config.get("speculators_config") or {}
    methods = spec.get("proposal_methods") if isinstance(spec, Mapping) else None
    tokens = methods[0].get("speculative_tokens") if isinstance(methods, list) and methods and isinstance(methods[0], Mapping) else None
    verifier = spec.get("verifier", {}).get("name_or_path") if isinstance(spec, Mapping) and isinstance(spec.get("verifier"), Mapping) else None
    key = "eagle_aux_hidden_state_layer_ids" if kind == EAGLE3 and "eagle_aux_hidden_state_layer_ids" in config else "aux_hidden_state_layer_ids"
    if key in config:
        taps = _ids(config.get(key), source, key)
    elif kind == EAGLE3:
        taps = _default_taps(target_layers, source, key)
    else:
        raise ConfigError(f"{source}: a block drafter needs {key}")
    return Drafter(kind, taps, _optional_int(layer, "hidden_size", source, 1), _optional_int(config, "target_hidden_size", source, 1),
                   _optional_int(config, "draft_vocab_size", source, 1), _optional_int(layer, "num_hidden_layers", source, 1),
                   _optional_int(config, "block_size", source, 2) if kind == BLOCK else None,
                   _optional_int(config, "mask_token_id", source),
                   _bool(config, "sample_from_anchor", source, False),
                   not _bool(config, "sliding_window_non_causal", source, True) if kind == BLOCK else True,
                   _bool(config, "enable_confidence_head", source, False),
                   None if tokens is None else _int(tokens, source, "speculative_tokens", 1),
                   verifier if isinstance(verifier, str) else None, "speculators:" + config["speculators_model_type"])


def read_drafter(config, source=CONFIG_NAME, target_layers=None):
    """The Drafter a parsed drafter config.json describes, or None when it is not a hidden-state drafter.

    target_layers (the target's decoder layer count) fills EAGLE-3's default taps and bounds every tap."""
    if not isinstance(config, Mapping):
        raise ConfigError(f"{source}: expected a JSON object")
    if target_layers is not None:
        _int(target_layers, source, "target_layers", 1)
    if "speculators_model_type" in config:
        drafter = _speculators(config, source, target_layers)
        return None if drafter is None else _checked(drafter, target_layers, source)
    crit = config.get("crit_drafter")
    if isinstance(crit, Mapping):
        if crit.get("kind") != MTP_HEAD:
            raise ConfigError(f"{source}: crit_drafter kind {crit.get('kind')!r} is not one this reader knows")
        layers = _int(crit.get("target_layers"), source, "crit_drafter.target_layers", 1)
        if target_layers is not None and layers != target_layers:
            raise ConfigError(f"{source}: the drafter was distilled from a {layers}-layer target, not this {target_layers}-layer one")
        target = crit.get("target") if isinstance(crit.get("target"), Mapping) else {}
        drafter = Drafter(MTP_HEAD, (layers,), _optional_int(config, "hidden_size", source, 1), None, None, 1, None, None,
                          False, True, False, None, target.get("path") if isinstance(target.get("path"), str) else None,
                          "crit:" + str(crit.get("format")))
        return _checked(drafter, target_layers, source)
    arch = _architectures(config)
    dflash = config.get("dflash_config")
    if isinstance(dflash, Mapping) or any("DSpark" in a for a in arch):
        section = dflash if isinstance(dflash, Mapping) else config
        if "target_layer_ids" not in section:
            raise ConfigError(f"{source}: a block drafter needs target_layer_ids")
        taps = tuple(i + 1 for i in _ids(section["target_layer_ids"], source, "target_layer_ids"))
        mask = section.get("mask_token_id", config.get("mask_token_id"))
        drafter = Drafter(BLOCK, taps, _optional_int(config, "hidden_size", source, 1), _optional_int(config, "target_hidden_size", source, 1),
                          _optional_int(config, "draft_vocab_size", source, 1), _optional_int(config, "num_hidden_layers", source, 1),
                          _optional_int(config, "block_size", source, 2),
                          None if mask is None else _int(mask, source, "mask_token_id"),
                          _bool(section, "sample_from_anchor", source, _bool(config, "sample_from_anchor", source, False)),
                          _bool(section, "causal", source, False), _bool(config, "enable_confidence_head", source, False),
                          None, None, "dflash" if isinstance(dflash, Mapping) else "dspark")
        return _checked(drafter, target_layers, source)
    if any("EAGLE3" in a.upper() for a in arch):
        section = config.get("eagle_config") if isinstance(config.get("eagle_config"), Mapping) else {}
        key = "eagle_aux_hidden_state_layer_ids"
        raw = section.get(key, config.get(key))
        taps = _ids(raw, source, key) if raw is not None else _default_taps(target_layers, source, key)
        drafter = Drafter(EAGLE3, taps, _optional_int(config, "hidden_size", source, 1), _optional_int(config, "target_hidden_size", source, 1),
                          _optional_int(config, "draft_vocab_size", source, 1), _optional_int(config, "num_hidden_layers", source, 1),
                          None, _optional_int(config, "mask_token_id", source), False, True, False, None, None, "eagle3")
        return _checked(drafter, target_layers, source)
    return None


def read_drafter_file(path, target_layers=None):
    """read_drafter over a drafter directory's config.json (or the file itself)."""
    path = Path(path)
    if path.is_dir():
        path = path / CONFIG_NAME
    try:
        config = jsonfile.read(path, max_bytes=MAX_CONFIG_BYTES, allow_nonfinite=True)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from None
    return read_drafter(config, str(path), target_layers)
