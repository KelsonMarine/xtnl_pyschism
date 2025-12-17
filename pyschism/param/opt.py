from datetime import datetime, timedelta
import logging

import pathlib
from typing import Union

import f90nml
import os
import re
from enum import Enum
import warnings

# import pytz

from pyschism import dates
from pyschism.mesh.fgrid import NchiType
from pyschism.param.utils import _extract_group_key_order, _to_fortran_scalar


logger = logging.getLogger(__name__)


# --- metaclass -------------------------------------------------------------


class OptMeta(type):
    def __new__(mcls, name, bases, attrs):
        default_path = pathlib.Path(__file__).parent / "param.nml"
        default_opt = f90nml.read(default_path)["opt"]

        # Schema: keys that are allowed (based on default template)
        attrs["_DEFAULT_TEMPLATE_PATH"] = default_path
        attrs["_DEFAULT_OPT_KEYS"] = list(default_opt.keys())  # may already be ordered
        attrs["_DEFAULT_OPT_DEFAULTS"] = dict(default_opt)

        # Predeclare attributes so hasattr()/getattr() behave as expected.
        # do not put a mutable list as a shared class default.
        for k, v in default_opt.items():
            attrs[k] = None

        return super().__new__(mcls, name, bases, attrs)


# --- OPT -------------------------------------------------------------------

class OPT(metaclass=OptMeta):
    """
    Read / write opt section of param.nml
    
    Provides [some] error checking implementation for OPT group
    
    Add error checking as you go.
    """

    def __init__(
        self,
        start_date: datetime = None,
        template: Union[os.PathLike, str, dict, f90nml.namelist.Namelist] = None,
        verbose: bool = True,
        **opt_kwargs,
    ):
        # (1) Load template
        template_src = template if template is not None else self._DEFAULT_TEMPLATE_PATH

        if isinstance(template_src, (dict, f90nml.namelist.Namelist)):
            opt_nml = template_src["opt"] if "opt" in template_src else template_src
            key_order = list(opt_nml.keys())
        else:
            template_path = pathlib.Path(template_src)
            nml = f90nml.read(template_path)
            opt_nml = nml["opt"]

            # Prefer true file order from raw scan; fallback to parsed order
            key_order = _extract_group_key_order(template_path, group="opt") or list(opt_nml.keys())

        self._template = dict(opt_nml)
        self._key_order = key_order

        # allowed keys = schema keys (from default) OR keys found in this template
        self._allowed_keys = set(self._DEFAULT_OPT_DEFAULTS.keys()) | set(self._template.keys())

        # Initialize instance attributes from template (copy lists!)
        for k in self._key_order:
            if k in self._template:
                v = self._template[k]
                if isinstance(v, (list, tuple)):
                    setattr(self, k, list(v))
                else:
                    setattr(self, k, v)

        # (2) Apply keywords provided except start_date (start_date handled via setter)
        if start_date is not None:
            self.start_date = start_date  

        # (3) Apply **opt_kwargs only if allowed; otherwise warn and ignore.
        for k, v in opt_kwargs.items():
            if k == "start_date":
                if v is not None:
                    self.start_date = v
                continue

            if k in self._allowed_keys and hasattr(self, k):
                setattr(self, k, v)
            else:
                warnings.warn(f"OPT: ignoring unknown option {k!r} (not in template/schema).", stacklevel=2)

    def __str__(self):
        # (4) Write Fortran namelist in template order with error checking
        lines = ["&OPT"]

        for k in self._key_order:
            if not hasattr(self, k):
                continue

            v = getattr(self, k)
            if v is None:
                continue

            # Basic sanity checks using the template value as a guide, if available
            v_tmpl = self._template.get(k, None)

            if isinstance(v, (list, tuple)):
                # Ensure list-like options remain list-like
                if v_tmpl is not None and not isinstance(v_tmpl, (list, tuple)):
                    raise TypeError(f"OPT.{k} should be a scalar (template is scalar), got list/tuple.")
                for i, item in enumerate(v, start=1):
                    if item is None:
                        continue
                    lines.append(f"\t{k}({i})={_to_fortran_scalar(item)}")
            else:
                if v_tmpl is not None and isinstance(v_tmpl, (list, tuple)):
                    raise TypeError(f"OPT.{k} should be a list (template is list), got {type(v)}.")
                lines.append(f"\t{k}={_to_fortran_scalar(v)}")

        lines.append("/")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        data: dict = {}

        for key in self._key_order:
            if not hasattr(self, key):
                continue

            current = getattr(self, key, None)

            # fall back to template default if unset
            if current is None:
                current = self._template.get(key, None)

            # if still None, skip
            if current is None:
                continue

            if isinstance(current, (list, tuple)):
                out = [0] * len(current)
                for i, state in enumerate(current):
                    # treat truthy values as 1, falsy as 0
                    out[i] = 1 if bool(state) else 0
                data[key] = out
            else:
                # optionally normalize boolean scalars to int (common in namelists)
                if isinstance(current, bool):
                    data[key] = int(current)
                else:
                    data[key] = current

        return data

    @property
    def start_date(self) -> Union[datetime, None]:
        return self._start_date

    @start_date.setter
    def start_date(self, start_date: Union[datetime, None]):
        if start_date is not None:
            start_date = dates.localize_datetime(start_date)
            self.start_year = start_date.year
            self.start_month = start_date.month
            self.start_day = start_date.day
            self.start_hour = start_date.hour
            self.start_hour += start_date.minute / 60.0
            self.utc_start = -start_date.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]  # noqa: E501
        else:
            self.start_year = None
            self.start_month = None
            self.start_day = None
            self.start_hour = None
            self.utc_start = None
        self._start_date = start_date

    @property
    def start_year(self) -> Union[int, None]:
        return self._start_year

    @start_year.setter
    def start_year(self, start_year: Union[int, None]):
        if start_year is not None:
            start_year = int(start_year)
        self._start_year = start_year

    @property
    def start_month(self) -> Union[int, None]:
        return self._start_month

    @start_month.setter
    def start_month(self, start_month: Union[int, None]):
        if start_month is not None:
            start_month = int(start_month)
        self._start_month = start_month

    @property
    def start_day(self) -> Union[int, None]:
        return self._start_day

    @start_day.setter
    def start_day(self, start_day: Union[int, None]):
        if start_day is not None:
            start_day = int(start_day)
        self._start_day = start_day

    @property
    def start_hour(self) -> Union[float, None]:
        return self._start_hour

    @start_hour.setter
    def start_hour(self, start_hour: Union[float, None]):
        if start_hour is not None:
            start_hour = float(start_hour)
        self._start_hour = start_hour

    @property
    def utc_start(self) -> Union[float, None]:
        return self._utc_start

    @utc_start.setter
    def utc_start(self, utc_start: Union[float, None]):
        if utc_start is not None:
            utc_start = float(utc_start)
        self._utc_start = utc_start

    @property
    def dramp(self) -> Union[float, None]:
        return self._dramp

    @dramp.setter
    def dramp(self, dramp: Union[float, timedelta, None]):
        if dramp is not None:
            if not isinstance(dramp, timedelta):
                dramp = timedelta(days=float(dramp))
            self._dramp = float(dramp / timedelta(days=1))
        else:
            self._dramp = None

    @property
    def nramp(self) -> Union[int, None]:
        if not hasattr(self, "_nramp") and self.dramp is not None:
            return 1
        elif hasattr(self, "_nramp"):
            return self._nramp

    @nramp.setter
    def nramp(self, nramp: Union[int, None]):
        if nramp not in [0, 1]:
            raise ValueError("Argument nramp must be 0, 1.")
        self._nramp = nramp

    @nramp.deleter
    def nramp(self):
        if hasattr(self, "_nramp"):
            del self._nramp

    @property
    def drampbc(self) -> Union[float, None]:
        return self._drampbc

    @drampbc.setter
    def drampbc(self, drampbc: Union[float, timedelta, None]):
        if drampbc is not None:
            if not isinstance(drampbc, timedelta):
                drampbc = timedelta(days=float(drampbc))
            self._drampbc = float(drampbc / timedelta(days=1))
        else:
            self._drampbc = None

    @property
    def nrampbc(self) -> Union[int, None]:
        if not hasattr(self, "_nrampbc") and self.drampbc is not None:
            return 1
        elif hasattr(self, "_nrampbc"):
            return self._nrampbc

    @nrampbc.setter
    def nrampbc(self, nrampbc: Union[int, None]):
        if nrampbc not in [0, 1]:
            raise ValueError("Argument nrampbc must be 0, 1.")
        self._nrampbc = nrampbc

    @nrampbc.deleter
    def nrampbc(self):
        if hasattr(self, "_nrampbc"):
            del self._nrampbc

    @property
    def if_source(self) -> Union[int, None]:
        if not hasattr(self, "_if_source"):
            return 0
        elif hasattr(self, "_if_source"):
            return self._if_source

    @if_source.setter
    def if_source(self, if_source: Union[int, None]):
        if if_source is not None:
            self._if_source = if_source
        else:
            self._if_source = None

    @property
    def dramp_ss(self) -> Union[float, None]:
        return self._dramp_ss

    @dramp_ss.setter
    def dramp_ss(self, dramp_ss: Union[float, timedelta, None]):
        if dramp_ss is not None:
            if not isinstance(dramp_ss, timedelta):
                dramp_ss = timedelta(days=float(dramp_ss))
            self._dramp_ss = float(dramp_ss / timedelta(days=1))
        else:
            self._dramp_ss = None

    @property
    def nramp_ss(self) -> Union[int, None]:
        if not hasattr(self, "_nramp_ss") and self.dramp_ss is not None:
            return 1
        elif hasattr(self, "_nramp_ss"):
            return self._nramp_ss

    @nramp_ss.setter
    def nramp_ss(self, nramp_ss: Union[int, None]):
        if nramp_ss not in [0, 1]:
            raise ValueError("Argument nramp_ss must be 0, 1.")
        self._nramp_ss = nramp_ss

    @nramp_ss.deleter
    def nramp_ss(self):
        if hasattr(self, "_nramp_ss"):
            del self._nramp_ss

    @property
    def drampwafo(self) -> Union[float, None]:
        return self._drampwafo

    @drampwafo.setter
    def drampwafo(self, drampwafo: Union[float, timedelta, None]):
        if drampwafo is not None:
            if not isinstance(drampwafo, timedelta):
                drampwafo = timedelta(days=float(drampwafo))
            self._drampwafo = float(drampwafo / timedelta(days=1))
        else:
            self._drampwafo = None

    @property
    def nrampwafo(self) -> Union[int, None]:
        if not hasattr(self, "_nrampwafo") and self.drampwafo is not None:
            return 1
        elif hasattr(self, "_nrampwafo"):
            return self._nrampwafo

    @nrampwafo.setter
    def nrampwafo(self, nrampwafo: Union[int, None]):
        if nrampwafo not in [0, 1]:
            raise ValueError("Argument nrampwafo must be 0, 1.")
        self._nrampwafo = nrampwafo

    @nrampwafo.deleter
    def nrampwafo(self):
        if hasattr(self, "_nrampwafo"):
            del self._nrampwafo

    @property
    def drampwind(self) -> Union[float, None]:
        return self._drampwind

    @drampwind.setter
    def drampwind(self, drampwind: Union[float, timedelta, None]):
        if drampwind is not None:
            if not isinstance(drampwind, timedelta):
                drampwind = timedelta(days=float(drampwind))
            self._drampwind = float(drampwind / timedelta(days=1))
        else:
            self._drampwind = None

    @property
    def nrampwind(self) -> Union[int, None]:
        if not hasattr(self, "_nrampwind") and self.drampwind is not None:
            return 1
        elif hasattr(self, "_nrampwind"):
            return self._nrampwind

    @nrampwind.setter
    def nrampwind(self, nrampwind: Union[int, None]):
        if nrampwind not in [0, 1]:
            raise ValueError("Argument nrampwind must be 0, 1.")
        self._nrampwind = nrampwind

    @nrampwind.deleter
    def nrampwind(self):
        if hasattr(self, "_nrampwind"):
            del self._nrampwind

    @property
    def nchi(self) -> Union[int, None]:
        return self._nchi

    @nchi.setter
    def nchi(self, nchi: Union[int, NchiType, None]):
        if not isinstance(nchi, NchiType) and nchi is not None:
            nchi = NchiType(nchi).value
        self._nchi = nchi

    @property
    def ic_elev(self) -> None:
        return self._ic_elev

    @ic_elev.setter
    def ic_elev(self, ic_elev: Union[int, None]):
        assert ic_elev in [0, 1, None]
        self._ic_elev = ic_elev

    @property
    def template(self):
        return self._template
    @template.setter
    def template(self, template):
        self._template = template

    # @property
    # def flag_ic(self) -> List[int]:
    #     if not hasattr(self, "_flag_ic"):
    #         self._flag_ic = len(FlagIcDescriptor.ic_types) * [0]
    #     return self._flag_ic


# --- I think none of this is needed anymore but kept for here just in case.

# class Dramp:

#     def __set__(self, obj, dramp: Union[int, float, timedelta, None]):

#         if not isinstance(dramp, (int, float, timedelta, type(None))):
#             raise TypeError("Argument drampbc must be an int, float, "
#                             "timedelta, or None.")

#         if isinstance(dramp, (int, float)):
#             dramp = timedelta(days=dramp)

#         if dramp is not None:
#             obj.nramp = 1

#         obj.__dict__['dramp'] = dramp

#     def __get__(self, obj, val):
#         return obj.__dict__.get('dramp')


# class Drampbc:

#     def __set__(self, obj, drampbc: Union[int, float, timedelta, None]):
#         if not isinstance(drampbc, (int, float, timedelta, type(None))):
#             raise TypeError("Argument drampbc must be an int, float, "
#                             "timedelta, or None.")

#         if isinstance(drampbc, (int, float)):
#             drampbc = timedelta(days=drampbc)

#         if drampbc is not None:
#             obj.nrampbc = 1

#         obj.__dict__['drampbc'] = drampbc

#     def __get__(self, obj, val):
#         return obj.__dict__.get('drampbc')


# class StartDate:

#     def __set__(self, obj, start_date: Union[datetime, None]):
#         if not isinstance(start_date, (datetime, type(None))):
#             raise TypeError("Argument start_date must be of type datetime or "
#                             "None. The datetime object will be assumed to be "
#                             "in UTC if ")
#         if start_date is not None:
#             if start_date.tzinfo is None \
#                     or start_date.tzinfo.utcoffset(start_date) is None:
#                 start_date = start_date.replace(tzinfo=pytz.utc)
#             obj.start_year = start_date.year
#             obj.start_month = start_date.month
#             obj.start_day = start_date.day
#             obj.start_hour = start_date.hour
#             obj.start_hour += start_date.minute / 60.
#             obj.utc_start = -start_date.utcoffset().total_seconds() / 3600  # type: ignore[union-attr]  # noqa: E501
#             # get rid of "negative" zero
#             obj.utc_start = +0. if obj.utc_start == -0. \
#                 else obj.utc_start
#         obj.__dict__['start_date'] = start_date

#     def __get__(self, obj, val):
#         return obj.__dict__.get('start_date')


# class Nchi:

#     def __set__(self, obj, fgrid):
#         nchi = fgrid.nchi
#         if nchi == -1:
#             obj.hmin_man = fgrid.hmin_man
#         if obj.nchi == 1:
#             obj.dbz_min = fgrid.dbz_min
#             obj.dbz_decay = fgrid.dbz_decay
#         obj.__dict__['nchi'] = nchi

#     def __get__(self, obj, val):
#         return obj.__dict__.get('nchi')


# class Ics:

#     def __set__(self, obj, ics: int):
#         obj.__dict__['ics'] = ics

#     def __get__(self, obj, val):
#         return obj.__dict__.get('ics')


# class Sfea0:

#     def __set__(self, obj, sfea0: float):
#         obj.__dict__['sfea0'] = sfea0

#     def __get__(self, obj, val):
#         return obj.__dict__.get('sfea0')


# class Ncor:

#     def __set__(self, obj, ncor: Coriolis):
#         if not isinstance(ncor, Coriolis):
#             raise TypeError(f"ncor must be of type {Coriolis}, not type "
#                             f"{type(ncor)}.")
#         obj.__dict__['ncor'] = ncor.value

#     def __get__(self, obj, val):
#         return obj.__dict__.get('ncor')


# class Ihot:

#     def __set__(self, obj, ihot: int):
#         if not isinstance(ihot, int):
#             raise TypeError(f"ihot must be of type {int}, not type "
#                             f"{type(ihot)}.")
#         if ihot not in [0, 1]:
#             raise ValueError("ihot must be 0 or 1")

#         obj.__dict__['ihot'] = ihot

#     def __get__(self, obj, val):
#         return obj.__dict__.get('ihot')


# class Nws:
#     def __set__(self, obj, nws: NWS):
#         if not isinstance(nws, NWS):
#             raise TypeError(
#                 f"nws must be of type {NWS}, not type {type(nws)}.")

#         if obj.start_date is None:
#             raise ValueError(
#                 "Can't initialize atmospheric data without start_date")
#             nws._start_date = obj.start_date

#         if isinstance(nws, NWS2):
#             obj.__dict__['nws'] = 2

#         else:
#             raise NotImplementedError(
#                 f'NWS type {type(nws)} is not implemented.')

#     def __get__(self, obj, val):
#         return obj.__dict__.get('nws')
