from pyschism.forcing.wwm.base import WWM  # Import waves first
from pyschism.forcing.wwm.base import WWM_IBOUNDFORMAT_1,WWM_IBOUNDFORMAT_3, get_wave_spr_DNV
from . import copernicus

__all__ = [
    'WWM',
    'WWM_IBOUNDFORMAT_1',
    'WWM_IBOUNDFORMAT_3',
    'get_wave_spr_DNV',
    # Optionally, if you want to allow users to import the submodule as a whole:
    'copernicus',
]