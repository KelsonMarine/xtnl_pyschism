from abc import ABC, abstractmethod
from pyschism.forcing.base import ModelForcing
import os
import pathlib
from typing import Union
from datetime import datetime, timedelta
import xarray as xr
import scipy.optimize
import warnings
import numpy as np

# ModelForcing is the base class.
# waves is an abstract subclass of ModelForcing.
# Parametric_Wave_Dataset is a concrete subclass of waves.

class WWM(ModelForcing):   
    """
    Abstract base class for wave datasets.

    Any subclass of waves must implement write() and dtype().
    This ensures that all subclasses follow a standard structure.
    Prevents Direct Instantiation

    waves cannot be instantiated directly, so users must create a subclass.
    This avoids accidentally creating an incomplete object.
    Encapsulates Shared Functionality

    If multiple subclasses (Parametric_Wave_Dataset, Spectral_Wave_Dataset, etc.) share common methods, those methods can be implemented in waves.
    
    For example, a shared write() implementation that subclasses can override

    """
    def __init__(self, wave_1 = None):  
        """Loads WaveDataset to use as wave input."""
        self.wave_1 = wave_1

    @abstractmethod
    def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
        """
        Provides a method for writing SCHISM wave files to disk.
        Must be implemented in subclasses.
        """
        pass  # Must be implemented in a subclass

    @property
    @abstractmethod
    def dtype(self):
        """Abstract property for wave dataset type."""
        pass

class WWM_IBOUNDFORMAT_1(WWM):
    """
    Parametric Wave Spectra at Boundary (LBCWA=T and LINHOM=F)

    WBHS           = 0.                 ! Hs at the boundary for parametric spectra
    WBSS           = 2.                 ! 1 or -1: Pierson-Moskowitz, 2 or -2: JONSWAP, 3 or -3: all in one BIN,
                                        ! 4: Gauss. The sign decides whether WBTP below is peak (+) or mean period (-)
    WBTP           = 8.                 ! Tp at the boundary (sec); mean or peak depending on the sign of WBSS
    WBDM           = 90.0               ! Avg. Wave Direction at the boundary
    WBDSMS         = 1.                 ! Directional spreading value in degrees (1) or as exponent (2)
    WBDS           = 20.                 ! Directioanl spreading at the boundary (degrees/exponent)
    WBGAUSS        = 0.1                ! factor for gaussian distribution if WBSS=1
    WBPKEN         = 3.3                ! Peak enhancement factor for Jonswap Spectra if WBSS=2 
    """
    def __init__(
            self, 
            WBHS: float = 0 ,
            WBSS: float = 2 ,
            WBTP: float = 8 , 
            WBDM: float = 0 ,
            WBDSMS: float = 1 ,
            WBDS: float = 20,
            WBGAUSS: float = 0.1,
            WBPKEN: float = 3.3
        ):
            super().__init__()  # Initialize the waves base class
            self.LBCWA = True
            self.LINHOM = False
            self.WBHS = WBHS
            self.WBSS = WBSS
            self.WBTP = WBTP
            self.WBDM = WBDM
            self.WBDSMS = WBDSMS
            self.WBDS = WBDS
            self.WBGAUSS = WBGAUSS
            self.WBPKEN = WBPKEN

    def write(self):
        """
        Implements the required write method.
        """
        
    def write(self):
        """
        Create a dictionary representing the wave boundary conditions
        in IBOUNDFORMAT=1 format with parametric wave spectra.
        """

        wwm_bouc = {
            'bouc': {
                'lbcse': False,     
                'lbinter': False,  
                'lbcwa': self.LBCWA, 
                'lbcsp': False,      
                'linhom': self.LINHOM, 
                'LBSP1D': False,
                'LBSP2D': False,
                'filebound': 'wwmbnd.gr3',  
                'iboundformat': 1,   
                'WBHS': self.WBHS,
                'WBSS': self.WBSS,
                'WBTP': self.WBTP,
                'WBDM': self.WBDM,
                'WBDSMS': self.WBDSMS,
                'WBDS': self.WBDS,
                'WBGAUSS': self.WBGAUSS,
                'WBPKEN': self.WBPKEN,
            }
        }
        
        return wwm_bouc
    
    @property
    def dtype(self):
        return "WWM_IBOUNDFORMAT_1"

class WWM_IBOUNDFORMAT_3(WWM):
    """
    Subclass implementing waves from parmeteric spectral statistics
    
    WWM IBOUNDFORMAT 3
    
    """
    def __init__(
        self, 
        ds: Union[xr.Dataset,None],
        resource: Union[str, os.PathLike, None]=None, 
        # hs_name: str ='hs', 
        # fp_name: str ='fp', 
        # t02_name: str ='t02',
        # dir_name: str ='dir', 
        # spr_name: str ='spr', 
        # dir_convention: str = 'Nautical',
    ):
        super().__init__()  # Initialize the waves base class
        self.resource = resource
        self.filewave = 'wwmbndfiles.dat'
        if ds is None:
            self.ds = xr.open_mfdataset(resource)
        else:
            self.ds = ds

    @property
    def dtype(self):
        return "WWM_IBOUNDFORMAT_3"

    def write(
            self, 
            outdir: Union[str, os.PathLike], 
            start_time: Union[np.ndarray,int,float] = None,
            rnday:  Union[np.ndarray,int,float] = None,
            end_time:  Union[np.ndarray,int,float] = None,
            bbox = None,
            overwrite: bool = True):
        """
        Implements the required write method.
        """
        
        path = pathlib.Path(outdir)
        path.mkdir(exist_ok=True)
        print(f"Writing Parametric_Wave_Dataset to {path}, overwrite={overwrite}")

        # Write each variable to a separate NetCDF file and record the filenames in wwm_inputs.txt.
        # Adjust the output directory if needed (here, files are written to the current directory).
        if os.path.isfile(path / self.filewave) and not overwrite:
            return (print(f'Did not write ww3_*.nc to file -- overwrite == False and files exist in {path}'))
        elif os.path.isfile(path / self.filewave) and overwrite: 
            os.remove(path / self.filewave)

        with open(path / self.filewave, "w") as f:
            for var in ['dir','fp','hs','spr','t02']: # order of variables matters (or so people claim?)
                filename = f"ww3_{var}.nc".strip()  # Ensure filename is clean.
                f.write(filename + "\n") 

        # Subset ds based on optional inputs
        if start_time is None and end_time is None and rnday is None:
            ds = self.ds
        else: # Slice in time
            if start_time is None:
                start_time = self.ds.time.isel(time=0).value
            elif end_time is None and rnday is not None:
                if not isinstance(rnday,timedelta):
                    rnday = timedelta(days=rnday)
                end_time = start_time + rnday
            elif rnday is None:
                rnday = end_time - start_time

            assert (end_time == start_time + rnday)

            assert (isinstance(type(self.ds.time.values), np.ndarray))
            ds = self.ds.sel(time=slice(start_time, end_time))

        # Slice in space
        if bbox is not None:
            ds = ds.sel(
                longitude=slice(bbox.xmin, bbox.xmax),
                latitude=slice(bbox.ymin, bbox.ymax)
            )

        # write to netcdf
        for var in ['dir','fp','hs','spr','t02']: # order of variables matters (or so people claim?)
                filename = f"ww3_{var}.nc".strip()  # Ensure filename is clean.
                print(f'writing: {path / filename}')    

                # extract variable
                ds_var = ds[var]

                # apply valid min and valid max
                valid_min = ds_var.attrs.get("valid_min", -np.inf)  # Default -inf if missing
                valid_max = ds_var.attrs.get("valid_max", np.inf)  # Default +inf if missing          
                ds_var = ds_var.where((ds_var >= valid_min) & (ds_var <= valid_max), np.nan) # Mask values outside valid range

                # transpose to expected dimension order from wwm
                # ds_var = ds_var.transpose("longitude", "latitude", "time") # this format does not work?
                ds_var = ds_var.transpose("time", "latitude","longitude") # this is the format given with examples of: ncdump -h ww3_hs.nc 

                # set encoding for WWM
                ds_var.encoding["dtype"]="int32"
                ds_var.encoding["scale_factor"]=0.002
                ds_var.encoding["_FillValue"]=-999999
                # ds_var.encoding["dtype"]="float32"
                # ds_var.encoding["scale_factor"]=1.0
                # ds_var.encoding["_FillValue"]=-999999

                # # compression
                # ds_var.encoding["zlib"]=True
                # ds_var.encoding["complevel"]=4

                # Define encoding for coordinates
                encoding = {
                    "longitude": {"dtype": "float64",'scale_factor': 1.},
                    "latitude": {"dtype": "float64",'scale_factor': 1.},
                    "time": {"dtype": "float64",'scale_factor': 1.},
                    var: ds_var.encoding  # Include variable-specific encoding
                }

                # print(encoding)

                # write to netcdf
                ds_var.to_netcdf(
                    path / filename,
                    unlimited_dims=['time'],
                    encoding=encoding,  # Ensures time, lon, lat are double
                    )
                
                # print(ds_var)
                


class Directional_Spectra_Wave_Dataset(WWM):
    """
    Subclass implementing waves from directional spectra boundary conditions
    """
    def __init__(
        self, 
        ds: Union[xr.Dataset,None],
        resource: Union[str, os.PathLike, None]=None, 
        # hs_name: str ='hs', 
        # fp_name: str ='fp', 
        # t02_name: str ='t02',
        # dir_name: str ='dir', 
        # spr_name: str ='spr', 
        # dir_convention: str = 'Nautical',
    ):
        super().__init__()  # Initialize the waves base class
        self.resource = resource
        self.filewave = 'wave_file'
        if ds is None:
            self.ds = xr.open_dataset(resource)
        else:
            self.ds

    @property
    def dtype(self):
        return "Directional_Spectra_Wave_Dataset"

    def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
        """
        Implements the required write method.
        """
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        print(f"Writing Directional_Spectra_Wave_Dataset to {path}, overwrite={overwrite}")

        # Example: Writing a NetCDF file
        self.ds.to_netcdf(path / f"{self.filewave}.nc", mode="w")


#########################################################################################
#
# Utility Functions
#
#########################################################################################

import xarray as xr
import numpy as np
from scipy.special import gamma
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt 
# # example call
# Tp_np = np.array([4.5, 7.0, 9.5, 15, 20])
# n_np, spr_estimate = get_directional_spreading_coef_DNV(Tp_np, showPlots=True)


def get_wave_spr_DNV(Tp: Union[list,np.array,xr.DataArray], showPlots=False):
    """
    Get cos-n spreading coefficient and angles based on wave period, based on DNV RP C205  

    Parameters
    ----------
    Tp : numpy.ndarray or xarray.DataArray
        Wave period values.
    showPlots : bool, optional
        If True, plot the interpolation function.
    
    Returns
    -------
    n : numpy.ndarray or xarray.DataArray
        The directional spreading coefficient, with the same shape (and
        coordinates/dimensions if Tp is an xarray.DataArray).
    spr_estimate : numpy.ndarray or xarray.DataArray
        An estimate of the mean directional spread (in degrees), computed using 
        spr_estimate_rad = sqrt(2/n).

    From: DNV-RP-C205, October 2010, Section 3.5.5.4
    """
    # Check if Tp is an xarray DataArray.
    is_xarray = hasattr(Tp, 'dims')
    
    # Get the underlying numpy values
    if is_xarray:
        Tp_vals = Tp.values
    else:
        Tp_vals = np.asarray(Tp)
    
    # Define anchor points for Tp and corresponding n values -- based on DNV
    Tp_breaks = np.array([0, 6, 8, 16, 25])
    n_breaks  = np.array([4, 4, 6, 8, 10])
    
    # Create a PCHIP interpolator.
    pchip_interp = PchipInterpolator(Tp_breaks, n_breaks, extrapolate=True)
    n_vals = pchip_interp(Tp_vals)
    
    # # Initialize output with NaNs (preserving shape)
    # n_vals = np.full(Tp_vals.shape, np.nan)
    
    # # Condition 1: wind sea, Tp < 6 => n = 4
    # mask = Tp_vals < 6
    # n_vals[mask] = 4
    
    # # Condition 2: wind sea, 6 <= Tp < 8 => n = 5
    # mask = (Tp_vals >= 6) & (Tp_vals < 8)
    # n_vals[mask] = 5
    
    # # Condition 3: swell, Tp > 8 => n = 6  (DNV says n>=6)
    # mask = Tp_vals > 8
    # n_vals[mask] = 6

    # # Condition 4: swell, Tp > 16 => n = 8
    # mask = Tp_vals > 16
    # n_vals[mask] = 8

    # If Tp was an xarray, wrap the output to preserve coordinates and dims.
    if is_xarray:
        n = xr.DataArray(n_vals, dims=Tp.dims, coords=Tp.coords, name='directional_spreading_coef')
    else:
        n = n_vals

    # Estimate directional spreading (in radians) using spr_estimate_rad = sqrt(2/n)
    spr_estimate_rad = np.sqrt(2 / n)
    spr_estimate = np.round(spr_estimate_rad * (180 / np.pi))  # convert to degrees
    
    if showPlots:
        def K(n):
            return gamma(n/2 + 1) / (np.sqrt(np.pi) * gamma(n/2 + 0.5))
        
        def cosN(n, theta):
            return np.cos(theta)**n
        
        def D(n, theta):
            return K(n) * cosN(n, theta)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f'D(θ) = K(n)·cosⁿ(θ)\ntp={Tp}\nn={n}\nspr={spr_estimate}')
        ax.set_xlabel('θ [rad]')
        ax.set_ylabel('D(θ)')
        
        theta = np.arange(-np.pi/2, np.pi/2, 0.01)
        unique_ns = np.unique(n_vals[~np.isnan(n_vals)]).astype(int)
        cmap = plt.get_cmap('winter', max(unique_ns)+1)

        # Loop over unique exponents. The MATLAB code plots only if n is even.
        for n_val in unique_ns:
            if n_val % 2 == 0:
                color = cmap(n_val)
                label = f'n={n_val}'
                ax.plot(theta, D(n_val, theta), '-', color=color, label=label)    

        ax.legend()
        ax.grid(True)
        plt.show()
    
    return n, spr_estimate


def gamma_from_Tp_DNV(Tp, Hs):
    x = Tp / np.sqrt(Hs)
    if x <= 3.6:
        return 5.0
    elif x < 5:
        return np.exp(5.75 - 1.15 * x)
    else:
        return 1.0

def Tm01_over_Tp_DNV(gamma):
    return 0.7303 + 0.049367 * gamma - 0.006556 * gamma**2 + 0.0003610 * gamma**3

def solve_Tp_gamma_from_Tm01_Hs_DNV(Tm01, Hs):

    def objective(Tp):
        gamma = gamma_from_Tp_DNV(Tp, Hs)
        phi = Tm01 - Tp * Tm01_over_Tp_DNV(gamma)
        return phi

    result = scipy.optimize.root_scalar(objective, bracket=[0.5, 30], method='brentq')
    if result.converged:
        Tp = result.root
        gamma = gamma_from_Tp_DNV(Tp, Hs)
        return Tp, gamma
    else:
        warnings.warn('pyschism.forcing.wwm.base.solve_Tp_gamma_from_Tm01_Hs_DNV failed -- using gamm=3.3 and .get_wave_Tp_from_Ta_DNV')
        gamma = 3.3
        Tp = get_wave_Tp_from_Ta_DNV(Tm01=Tm01,gamma=gamma)
        return Tp, gamma


def get_wave_Tp_from_Ta_DNV(Tm01: Union[float, np.ndarray], gamma: Union[float, np.ndarray] = 3.3) -> np.ndarray:
    """
    Estimate wave peak period (Tp) from wave mean period (Ta) and gamma,
    using DNV-GL empirical relationship for JONSWAP spectrum.

    Parameters:
        Tm01 : float or np.ndarray
            Mean wave period (Ta)
        gamma : float or np.ndarray, optional
            JONSWAP peak enhancement factor (default is 3.3)

    Returns:
        Tp : np.ndarray
            Estimated peak wave period (Tp)

    From: DNV-RP-C205, October 2010, Section 3.5.5.4
    """
    Tm01 = np.asarray(Tm01, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if gamma.size != 1 and Tm01.shape != gamma.shape:
        raise ValueError("Tm01 and gamma must have the same shape or gamma must be a scalar.")

    denom = Tm01_over_Tp_DNV(gamma)
    Tp = Tm01 / denom

    return Tp

