from pyschism.forcing.wwm.base import WWM  # Import waves first
from pyschism.forcing.wwm.base import Parametric_Wave_Dataset, get_wave_spr_DNV
from . import copernicus

__all__ = [
    'WWM',
    'Parametric_Wave_Dataset',
    'Directional_Wave_Dataset',
    'get_wave_spr_DNV',
    # Optionally, if you want to allow users to import the submodule as a whole:
    'copernicus',
]