"""The norms slot's plain plugin: fused-norms' FP64 rows (rows.py) as the host form and its torch ops (torch_norms.py)
as the device form."""
from .norms import rows, torch_norms


class TorchNorms(rows.Rows):
    pass


for _name in torch_norms.OPERATIONS:
    setattr(TorchNorms, _name, staticmethod(getattr(torch_norms, _name)))


def create(context, options, parts):
    return TorchNorms()
