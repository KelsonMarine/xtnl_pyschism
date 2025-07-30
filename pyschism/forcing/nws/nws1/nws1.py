
import os
import pathlib
from typing import Union
import numpy as np
import f90nml  # type: ignore[import]

from pyschism.enums import NWSType
from pyschism.forcing.nws.base import NWS
from pyschism.forcing.nws.nws1.windth import WindthDataset
from pyschism.mesh import gridgr3


SFLUX_DEFAULTS = f90nml.read(pathlib.Path(__file__).parent / "sflux_inputs.txt")

class NWS1(NWS):
    def __init__(
        self,
        windth: WindthDataset = None,
        windrot: gridgr3.Windrot = None,
    ):
        """Loads WindthDataset to use as NWS1 input."""

        self.windth = windth
        self.windrot = windrot

    def __str__(self):
        data = []
        data = "\n".join(data)
        return f"&sflux_inputs\n{data}/\n"

    @classmethod
    def read(cls, path, windth_glob="*_1.*"):
        path = pathlib.Path(path)
        return cls(
            windth=WindthDataset(list(path.glob(windth_glob))),
        )

    def write(
        self,
        path: Union[str, os.PathLike],
        start_date=None,
        end_date=None,
        overwrite: bool = False,
        windrot: bool = True
    ):

        # write sflux namelist
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=overwrite)
        # with open(path / "sflux_inputs.txt", "w") as f:
        #     f.write(str(self))
        
        # Build time series
        dT_days = 1
        times = np.arange(0, rnday + dT_days, dT_days) * 86400
        data = np.column_stack([times, np.full_like(self.times, self.u10), np.full_like(times, self.v10)])

        # Write to ASCII file
        outdir.mkdir(exist_ok=True)
        with open(outdir / filename, 'w') as f:
            for t, u, v in data:
                f.write(f"{t:.5f} {u:.5f} {v:.5f}\n")

        print(f"✅ File written: {outdir / filename}")



        # # write windrot data
        if windrot is not False and self.windrot is not None:
            windrot = "windrot_geo2proj.gr3" if windrot is True else windrot
            self.windrot.write(path.parent / "windrot_geo2proj.gr3", overwrite)

    @property
    def dtype(self) -> NWSType:
        """Returns the datatype of the object"""
        return NWSType(2)

    @property
    def windth(self) -> WindthDataset:
        return self._windth

    @windth.setter
    def windth(self, windth):
        if not isinstance(windth, WindthDataset):
            raise TypeError(
                f"Argument windth must be of type {WindthDataset},"
                f" not type {type(windth)}"
            )
        self._windth = windth

    @property
    def sflux_2(self) -> WindthDataset:
        return self._sflux_2

    @sflux_2.setter
    def sflux_2(self, sflux_2):
        if sflux_2 is not None:
            if not isinstance(sflux_2, WindthDataset):
                raise TypeError(
                    f"Argument sflux_2 must be of type {WindthDataset}, not "
                    f"type {type(sflux_2)}."
                )
        self._sflux_2 = sflux_2

    @property
    def windrot(self):
        return self._windrot

    @windrot.setter
    def windrot(self, windrot: Union[gridgr3.Windrot, None]):
        if not isinstance(windrot, (gridgr3.Windrot, type(None))):
            raise TypeError(
                f"Argument windrot must be of type {gridgr3.Windrot} or None, "
                f"not type {type(windrot)}."
            )
        self._windrot = windrot
