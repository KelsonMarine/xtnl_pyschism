from abc import ABC, abstractmethod
from pyschism.forcing.base import ModelForcing
import os
import pathlib
from typing import Union
from datetime import datetime, timedelta
import xarray as xr


# ModelForcing is the base class.
# waves is an abstract subclass of ModelForcing.
# Parametric_Wave_Dataset is a concrete subclass of waves.

class waves(ModelForcing):   
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
    def __init__(self, wave_1: Union["waves", None] = None):  # Forward reference
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


class Parametric_Wave_Dataset(waves):
    """
    Subclass implementing waves.
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
        return "Parametric_Wave_Dataset"

    def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
        """
        Implements the required write method.
        """
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        print(f"Writing Parametric_Wave_Dataset to {path}, overwrite={overwrite}")

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
# waveTp_np = np.array([4.5, 7.0, 9.5, 15, 20])
# n_np, spr_estimate = get_directional_spreading_coef_DNV(waveTp_np, showPlots=True)


def get_wave_spr_DNV(waveTp, showPlots=False):
    """
    Get cos-n spreading coefficient and angles based on wave period, based on DNV RP C205  

    Parameters
    ----------
    waveTp : numpy.ndarray or xarray.DataArray
        Wave period values.
    showPlots : bool, optional
        If True, plot the interpolation function.
    
    Returns
    -------
    n : numpy.ndarray or xarray.DataArray
        The directional spreading coefficient, with the same shape (and
        coordinates/dimensions if waveTp is an xarray.DataArray).
    spr_estimate : numpy.ndarray or xarray.DataArray
        An estimate of the mean directional spread (in degrees), computed using 
        spr_estimate_rad = sqrt(2/n).
    """
    # Check if waveTp is an xarray DataArray.
    is_xarray = hasattr(waveTp, 'dims')
    
    # Get the underlying numpy values
    if is_xarray:
        waveTp_vals = waveTp.values
    else:
        waveTp_vals = np.asarray(waveTp)
    
    # Define anchor points for Tp and corresponding n values -- based on DNV
    Tp_breaks = np.array([0, 6, 8, 16, 25])
    n_breaks  = np.array([4, 4, 6, 8, 10])
    
    # Create a PCHIP interpolator.
    pchip_interp = PchipInterpolator(Tp_breaks, n_breaks, extrapolate=True)
    n_vals = pchip_interp(waveTp_vals)
    
    # # Initialize output with NaNs (preserving shape)
    # n_vals = np.full(waveTp_vals.shape, np.nan)
    
    # # Condition 1: wind sea, Tp < 6 => n = 4
    # mask = waveTp_vals < 6
    # n_vals[mask] = 4
    
    # # Condition 2: wind sea, 6 <= Tp < 8 => n = 5
    # mask = (waveTp_vals >= 6) & (waveTp_vals < 8)
    # n_vals[mask] = 5
    
    # # Condition 3: swell, Tp > 8 => n = 6  (DNV says n>=6)
    # mask = waveTp_vals > 8
    # n_vals[mask] = 6

    # # Condition 4: swell, Tp > 16 => n = 8
    # mask = waveTp_vals > 16
    # n_vals[mask] = 8

    # If waveTp was an xarray, wrap the output to preserve coordinates and dims.
    if is_xarray:
        n = xr.DataArray(n_vals, dims=waveTp.dims, coords=waveTp.coords, name='directional_spreading_coef')
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
        ax.set_title(f'D(θ) = K(n)·cosⁿ(θ)\ntp={waveTp}\nn={n}\nspr={spr_estimate}')
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
