from datetime import timedelta
# from enum import Enum
import pathlib
import os
from typing import Union

import f90nml

import f90nml.namelist
from pyschism.enums import Stratification
from pyschism.param.utils import _extract_group_key_order, _to_fortran_scalar

# class IbcType(Enum):
#     BAROCLINIC = 0
#     BAROTROPIC = 1

#     @classmethod
#     def _missing_(self, name):
#         raise ValueError(f'{name} is not a valid integer for ibc. '
#                          'Valid integers are 0 or 1.')

class CoreMeta(type):
    def __new__(mcls, name, bases, attrs):
        default_path = pathlib.Path(__file__).parent / "param.nml"
        default_opt = f90nml.read(default_path)["core"]

        # Schema: keys that are allowed (based on default template)
        attrs["_DEFAULT_TEMPLATE_PATH"] = default_path
        attrs["_DEFAULT_OPT_KEYS"] = list(default_opt.keys())  # may already be ordered
        attrs["_DEFAULT_OPT_DEFAULTS"] = dict(default_opt)

        # Predeclare attributes so hasattr()/getattr() behave as expected.
        for k, v in default_opt.items():
            attrs[k] = None

        return super().__new__(mcls, name, bases, attrs)


class CORE(metaclass=CoreMeta):
    """Provides error checking implementation for CORE group"""

    mandatory = ["ipre", "ibc", "ibtp", "rnday", "dt", "nspool", "ihfskip"]

    def __init__(
        self,
        ipre: int = 0,
        ibc: int = 0,
        ibtp: int = 0,
        rnday: Union[float, timedelta] = 0.0,
        dt: Union[float, timedelta] = 150.0,
        nspool: Union[int, float, timedelta] = None,
        ihfskip: Union[int, timedelta] = None,
        nmarsh_types: int = 1,
        nbins_veg_vert: int = 1,
        ntracer_gen: int = 2,
        ntracer_age: int = 4,
        sed_class: int = 5, 
        eco_class: int = 27,
        template: Union[bool, str, os.PathLike, dict, f90nml.namelist.Namelist] = None,
        verbose: bool = True
    ):
        

        # (1) Load template
        template_src = template if template is not None else self._DEFAULT_TEMPLATE_PATH

        if isinstance(template_src, (dict, f90nml.namelist.Namelist)):
            opt_nml = template_src["core"] if "core" in template_src else template_src
            key_order = list(opt_nml.keys())
        else:
            template_path = pathlib.Path(template_src)
            nml = f90nml.read(template_path)
            opt_nml = nml["core"]

            # Prefer true file order from raw scan; fallback to parsed order
            key_order = _extract_group_key_order(template_path, group="core") or list(opt_nml.keys())

        self._template = dict(opt_nml)
        self._key_order = key_order

        # allowed keys = schema keys (from default) OR keys found in this template
        self._allowed_keys = set(self._DEFAULT_OPT_DEFAULTS.keys()) | set(self._template.keys())

        self.ipre = ipre
        self.ibc = ibc
        self.ibtp = ibtp
        self.rnday = rnday
        self.dt = dt
        self.nspool = nspool
        self.ihfskip = ihfskip
        self.ntracer_gen = ntracer_gen
        self.nmarsh_types = nmarsh_types 
        self.nbins_veg_vert = nbins_veg_vert
        self.nmarsh_types = nmarsh_types
        self.ntracter_age = ntracer_age
        self.sed_class = sed_class
        self.eco_class = eco_class
        self.verbose = verbose

    def __str__(self):
        lines = []
        for k in self._key_order:
            if not hasattr(self, k):
                continue
            v = getattr(self, k)
            if v is None:
                continue
            lines.append(f"\t{k}={_to_fortran_scalar(v)}")
        lines = "\n".join(lines)
        return f"&CORE\n{lines}\n/"


    def to_dict(self):
        output = {}
        for k in self._key_order:
            if not hasattr(self, k):
                continue
            v = getattr(self, k)
            if v is None:
                continue
            output[k]=v
        return output

    @property
    def ipre(self) -> int:
        return self._ipre

    @ipre.setter
    def ipre(self, ipre: int):
        if ipre not in [0, 1]:
            raise ValueError("Argument to ipre attribute must be 0 or 1")
        self._ipre = ipre

    @property
    def ibc(self):
        return self._ibc

    @ibc.setter
    def ibc(self, ibc: Union[Stratification, int, str]):

        if isinstance(ibc, str):
            ibc = Stratification[ibc.upper()].value

        if isinstance(ibc, Stratification):
            ibc = ibc.value

        if isinstance(ibc, int):
            if ibc not in [0, 1]:
                raise ValueError(
                    "Argument to attribute ibc must be of type "
                    f"{Stratification} or an 0, 1 integer or a string "
                    "'barotropic', 'baroclinic', not type "
                    f"{type(ibc)}."
                )

        self._ibc = ibc

    @property
    def ibtp(self):
        return self._ibtp

    @ibtp.setter
    def ibtp(self, ibtp: int):

        if ibtp not in [0, 1]:
            raise TypeError(
                "Argument to attribute ibtp must be 0 or 1, not " f"{ibtp}."
            )
        if ibtp == 1 and self.ibc == 0:
            raise ValueError("ibtp cannot be set to 1 because ibc is equal to " "zero.")
        self._ibtp = ibtp

    @property
    def rnday(self) -> Union[float, None]:
        return self._rnday

    @rnday.setter
    def rnday(self, rnday: Union[float, timedelta, None]):
        if rnday is not None:
            if not isinstance(rnday, timedelta):
                rnday = timedelta(days=float(rnday))
            self._rnday = rnday / timedelta(days=1)
        else:
            self._rnday = None

    @property
    def dt(self) -> Union[float, None]:
        return self._dt

    @dt.setter
    def dt(self, dt: Union[float, timedelta, None]):
        if dt is None:
            dt = timedelta(seconds=150.0)

        if not isinstance(dt, timedelta):
            dt = timedelta(seconds=float(dt))

        self._dt = dt.total_seconds()

    @property
    def nspool(self):
        return self._nspool

    @nspool.setter
    def nspool(self, nspool: Union[int, float, timedelta, None]):
        if nspool is None and self.rnday is not None:
            nspool = int(round(self.rnday / self.dt))
        if isinstance(nspool, timedelta):
            nspool = int(round(nspool.total_seconds() / self.dt))
        if isinstance(nspool, float):
            nspool = int(round(timedelta(hours=nspool).total_seconds() / self.dt))
        if isinstance(nspool, (int, float)):
            if nspool < 0:
                raise ValueError("nspool must be positive.")
        if nspool is not None:
            self._nspool = int(nspool)
        else:
            self._nspool = None

    @property
    def ihfskip(self):
        return self._ihfskip

    @ihfskip.setter
    def ihfskip(self, ihfskip: Union[int, timedelta, None]):

        if not isinstance(ihfskip, (int, timedelta, type(None))):
            raise TypeError("Argument ihfskip must be int, timedelta or None.")

        if ihfskip is None and self.rnday is not None:
            ihfskip = int(round(timedelta(days=self.rnday).total_seconds() / self.dt))

        if isinstance(ihfskip, timedelta):
            ihfskip = int(round(ihfskip.total_seconds() / self.dt))

        if isinstance(self.nspool, int):
            if self.nspool > 0:
                if not (ihfskip / self.nspool).is_integer():
                    raise ValueError(
                        "ihfskip/nspool must be an integer but got "
                        "ihfskip/nspool="
                        f"{ihfskip}/{self.nspool}={ihfskip/self.nspool}"
                    )

        self._ihfskip = ihfskip

    @property
    def nmarsh_types(self):
        return self._nmarsh_types

    @nmarsh_types.setter
    def nmarsh_types(self, nmarsh_types:int):
        self._nmarsh_types = nmarsh_types

    @property
    def nbins_veg_vert(self):
        return self._nbins_veg_vert

    @nbins_veg_vert.setter
    def nbins_veg_vert(self, nbins_veg_vert:int):
        self._nbins_veg_vert = nbins_veg_vert

    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, template: Union[bool, str, os.PathLike, dict, f90nml.Namelist ,None]):
        if template is True:
            template = f90nml.read(pathlib.Path(__file__) / 'param.nml')
        elif template is False or template is None:
            template = None
        elif isinstance(template,str) or isinstance(template,os.PathLike):
            template =  f90nml.read(template)['core']
        elif isinstance(template,f90nml.Namelist):
            template = template['core']
        elif isinstance(template, dict): 
            template = template
        self._template = template

    @property
    def defaults(self):
        if self.template is None:
            return f90nml.read(self.template)["core"]
        return self._template