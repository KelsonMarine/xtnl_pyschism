from datetime import timedelta
import logging
from typing import Union
import os
import f90nml
import pathlib
import warnings
from pyschism.enums import (
    # IofWetdryVariables,
    # IofZcorVariables,
    IofHydroVariables,
    IofDvdVariables,
    IofWwmVariables,
    IofGenVariables,
    IofAgeVariables,
    IofSedVariables,
    IofEcoVariables,
    IofIcmVariables,
    IofCosVariables,
    IofFibVariables,
    IofSed2dVariables,
    IofMarshVariables,
    IofIceVariables,
    IofAnaVariables,
    SchoutType
)
from pyschism.param.utils import _extract_group_key_order, _to_fortran_scalar


_logger = logging.getLogger(__name__)


class SurfaceOutputVars:

    def __init__(self):
        self._surface_output_vars = {
            'iof_hydro': [(var.value, i) for i, var
                          in enumerate(IofHydroVariables)],
            'iof_wwm': [(var.value, i) for i, var
                        in enumerate(IofWwmVariables)],
            'iof_dvd': [(var.value, i) for i, var
                        in enumerate(IofDvdVariables)],
            'iof_gen': [(var.value, i) for i, var
                        in enumerate(IofGenVariables)],
            'iof_age': [(var.value, i) for i, var
                        in enumerate(IofAgeVariables)],
            'iof_sed': [(var.value, i) for i, var
                        in enumerate(IofSedVariables)],
            'iof_eco': [(var.value, i) for i, var
                        in enumerate(IofEcoVariables)],
            'iof_icm': [(var.value, i) for i, var
                        in enumerate(IofIcmVariables)],
            'iof_cos': [(var.value, i) for i, var
                        in enumerate(IofCosVariables)],
            'iof_fib': [(var.value, i) for i, var
                        in enumerate(IofFibVariables)],
            'iof_sed2d': [(var.value, i) for i, var
                          in enumerate(IofSed2dVariables)],
            'iof_marsh': [(var.value, i) for i, var
                          in enumerate(IofMarshVariables)],
            'iof_ice': [(var.value, i) for i, var
                        in enumerate(IofIceVariables)],
            'iof_ana': [(var.value, i) for i, var
                        in enumerate(IofAnaVariables)],
        }

    def __get__(self, obj, val):
        return self._surface_output_vars


class OutputVariableDescriptor:

    def __init__(self, iof_type, name, index):
        self._iof_type = iof_type
        self._name = name
        self._index = index

    def __get__(self, obj, val):
        return bool(getattr(obj, f'_{self._iof_type}')[self._index])

    def __set__(self, obj, val: bool):
        if not isinstance(val, bool):
            raise TypeError(f'Argument to {self._name} must be boolean, not '
                            f'type {type(val)}.')
        iof = getattr(obj, f'_{self._iof_type}')
        iof[self._index] = int(val)


class SchoutMeta(type):

    surface_output_vars = SurfaceOutputVars()

    def __new__(meta, name, bases, attrs):
        for iof_type in meta.surface_output_vars.keys():
            attrs[f'_{iof_type}'] = len(SchoutType[iof_type].value)*[0]

        for iof_type, vardata in meta.surface_output_vars.items():
            for name, index in vardata:
                attrs[name] = OutputVariableDescriptor(iof_type, name, index)
        output_vars = []
        for iof_, outputs in meta.surface_output_vars.items():
            for name, id in outputs:
                output_vars.append(name)
        # attrs['surface_output_vars'] = output_vars
        attrs['surface_output_var_names'] = output_vars
        attrs['surface_output_vars_map'] = meta.surface_output_vars  # keep the dict
        return type(name, bases, attrs)


class SCHOUT(
        metaclass=SchoutMeta
):
    """ Provides error checking implementation for SCHOUT group """

    def __init__(
            self,
            nc_out: Union[bool, int] = True,
            iof_ugrid: int = 2,
            nhot: Union[bool, int] = True,
            nhot_write: int = None,
            iout_sta: Union[bool, int] = False,
            nspool_sta: int = None,
            template: Union[bool, str, os.PathLike, dict, f90nml.namelist.Namelist] = None,
            verbose: bool = True,
            **outputs
    ):
        """
        nhot_write:
            - if -1 will write last timestep (default)
            - if None it will be disabled.
            - if int interpreted as iteration
            - if timedelta it will be rounded to the nearest iteration
        template: template param.nml containing &schout section

        **outputs: name-value pairs of 
        """

        # copy class-default list into the instance for each 
        for iof_type in self.__class__.surface_output_vars_map.keys():
            setattr(self, f'_{iof_type}', getattr(self.__class__, f'_{iof_type}').copy())

        # !-----------------------------------------------------------------------
        # ! Main switch to control netcdf. If =0, SCHISM won't output nc files 
        # ! at all (useful for other programs like ESMF to output)
        # !-----------------------------------------------------------------------
        self.nc_out = nc_out

        # !-----------------------------------------------------------------------
        # ! UGRID option for _3D_ outputs under scribed IO (out2d*.nc always has meta
        # ! data info). If iof_ugrid > 0, 3D outputs will also have UGRID metadata.  
        # ! if iof_ugrid == 1 3D output will contain UGRID mesh data (at the expense
        # ! of file size); if iof_ugrid == 2, 3D output references mesh data in the 
        # ! 2D output.
        # !-----------------------------------------------------------------------
        self.iof_ugrid = iof_ugrid

        # !-----------------------------------------------------------------------
        # ! Option for hotstart outputs
        # !-----------------------------------------------------------------------
        self.nhot = nhot # !0 : no *_hotstart.nc,  1: output *_hotstart.nc every 'nhot_write' steps
        self.nhot_write = nhot_write # !must be a multiple of ihfskip if nhot=1 (output *_hotstart every 'nhot_write' time steps)

        # !-----------------------------------------------------------------------
        # ! Station output option. If iout_sta/=0, need output skip (nspool_sta) and
        # ! a station.in. If ics=2, the cordinates in station.in must be in lon., lat,
        # ! and z (positive upward; not used for 2D variables). 
        # !-----------------------------------------------------------------------
        self.iout_sta = iout_sta
        self.nspool_sta = nspool_sta # !needed if iout_sta/=0; mod(nhot_write,nspool_sta) must=0

        self.verbose = verbose
        template = template if template is not None else pathlib.Path(__file__).parent / 'param.nml'

        # Set template values
        self.template = f90nml.read(template)["schout"]
        # for key, value in self.template.items():
        #     if getattr(self,key,None) is None:
        #         setattr(self,key,value)
        for key, value in self.template.items():
            if key.startswith("iof_") and isinstance(value, (list, tuple)):
                setattr(self, f'_{key}', list(value))
            else:
                if getattr(self, key, None) is None:
                    setattr(self, key, value)

        # Set individual outputs via kwargs
        for key, val in outputs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                warnings.warn(f"SCHOUT: ignoring unknown output {key!r}.", stacklevel=2)

    def __iter__(self):
        for outvar in self._surface_output_vars:
            yield outvar, getattr(self, outvar)

    def __str__(self):
        schout = ["&SCHOUT"]
        for k in ['nc_out', 'iof_ugrid','nhot','nhot_write','iout_sta','nspool_sta']:
            v = getattr(self,k,None)
            if v is None:
                continue
            schout.append(f"\t{k}={_to_fortran_scalar(v)}")
        for k in self.surface_output_vars_map.keys():
            for i, state in enumerate(getattr(self, f'_{k}')):
                if self.verbose or state:
                    schout.append(f'\t{k}({i+1})={_to_fortran_scalar(state)}')
        schout.append('/')
        return '\n'.join(schout)

    def to_dict(self):
        data = {}
        if self.nhot_write is not None:
            data['nhot'] = self.nhot
            data['nhot_write'] = self.nhot_write
        if self.nspool_sta is not None:
            nspool_sta = self.nspool_sta
            if isinstance(nspool_sta, timedelta):
                nspool_sta = int(round(nspool_sta.total_seconds() / self.dt))
            if isinstance(nspool_sta, float):
                nspool_sta = int(
                    round(timedelta(hours=nspool_sta) / self.dt))
            if isinstance(nspool_sta, (int, float)):
                if nspool_sta <= 0:
                    raise ValueError("nspool_sta must be positive.")
            data['iout_sta'] = self.iout_sta
            data['nspool_sta'] = nspool_sta
        for var in dir(self):
            if var.startswith('_iof'):
                _var = var[1:]
                for i, state in enumerate(getattr(self, var)):
                    # if state == 1:
                    #     data[_var][i] = state
                    data[_var][i] = state
        return data
    
    
    def to_dict(self):
        schout = {}
        schout = ["&SCHOUT"]
        for k in ['nc_out', 'iof_ugrid','nhot','nhot_write','iout_sta','nspool_sta']:
            v = getattr(self,k,None)
            if v is None:
                continue
            schout[k] = v
        for k in self.surface_output_vars_map.keys():
            schout[k] = getattr(self, k)
        return schout

    @property
    def nhot_write(self):
        return self._nhot_write

    @nhot_write.setter
    def nhot_write(self, nhot_write: Union[int, None]):
        if nhot_write is not None:
            if not isinstance(nhot_write, int):
                raise TypeError(
                    f'Argument nhot_write must be of type {int} or None, not '
                    f'type {type(nhot_write)}.')
        self._nhot_write = nhot_write

    @property
    def nhot(self) -> Union[int, None]:
        if not hasattr(self, '_nhot') and self.nhot_write is not None:
            return 1
        else:
            return self._nhot

    @nhot.setter
    def nhot(self, nhot: Union[int, None]):
        if nhot not in [0, 1]:
            raise ValueError('Argument nhot must be 0, 1.')
        self._nhot = nhot

    @nhot.deleter
    def nhot(self):
        if hasattr(self, '_nhot'):
            del self._nhot

    @property
    def nspool_sta(self):
        return self._nspool_sta

    @nspool_sta.setter
    def nspool_sta(self, nspool_sta: Union[int, None]):
        if nspool_sta is not None:
            if not isinstance(nspool_sta, int):
                raise TypeError(
                    f'Argument nspool_sta must be of type {int} or None, not '
                    f'type {type(nspool_sta)}.')
        self._nspool_sta = nspool_sta

    @property
    def iout_sta(self) -> Union[int, None]:
        if not hasattr(self, '_iout_sta') and self.nspool_sta is not None:
            return 1
        else:
            return self._iout_sta

    @iout_sta.setter
    def iout_sta(self, iout_sta: Union[int, None]):
        if iout_sta not in [0, 1]:
            raise ValueError('Argument iout_sta must be 0, 1.')
        self._iout_sta = iout_sta

    @iout_sta.deleter
    def iout_sta(self):
        if hasattr(self, '_iout_sta'):
            del self._iout_sta
