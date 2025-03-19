import xarray as xr
import copernicusmarine
import pyschism.mesh.hgrid 
import wavespectra
import os, pathlib 
import subprocess
from datetime import timedelta, datetime, timezone
import numpy as np
from pyschism.forcing.wwm.base import waves, Parametric_Wave_Dataset, get_wave_spr_DNV
from typing import Union, Literal

class GLOBAL_MULTIYEAR_WAV_001_032(Parametric_Wave_Dataset):

    """
    Subset Copernicus Marine Wave Reanalysis Data: GLOBAL_MULTIYEAR_WAV_001_032
    """

    def __init__(
        self,
        ds: Parametric_Wave_Dataset = None,
        dataset_id: Literal['cmems_mod_glo_wav_my_0.2deg_PT3H-i', 'cmems_mod_glo_wav_myint_0.2deg_PT3H'] =  'cmems_mod_glo_wav_myint_0.2deg_PT3H',
        username: str = None,
        password: str = None
    ):
    
        """Loads Parametric_Wave_Dataset to use as waves input."""
        self.dataset_id = dataset_id
        self.filewave = 'bndfiles.dat'
        if waves is not None:
            self.ds = ds
        else:
            if username is not None and password is not None:            
                copernicusmarine.login(username=username,password=password,force_overwrite=True)
            
    def describe(self):
        #'GLOBAL_MULTIYEAR_WAV_001_032'
        # dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i' # 15/01/1980 to 30/04/2023
        # dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H'  # interim period, 01/05/2023 to present (month - 1)
        dataset_describe = copernicusmarine.describe(contains=[self.dataset_id])
        description=dataset_describe.model_dump()
        return description

    def write(
        self,
        outdir: Union[str, os.PathLike],
        start_datetime: datetime = None,
        rnday: Union[float, int, timedelta] = None,
        end_datetime: datetime = None,
        bbox: Union[list, tuple] = None,
        # overwrite: bool = True, # TODO: add error catching here (always overwrites!)
        # cleanup: bool = True
    ):
        
        if end_datetime is None:
            if not isinstance(rnday,timedelta):
                rnday = timedelta(days=rnday)
            end_datetime = start_datetime + rnday
        
        # define a dict for later renaming
        variables = {
            "VHM0":"hs", # Sea surface wave significant height [m]
            "VTPK":"tp", # Sea surface wave period at variance spectral density maximum [s]
            "VTM02":"t02", # Sea surface wave mean period from variance spectral density second frequency moment [s]
            "VMDR":"dir", # Sea surface wave from direction [degrees]
        }

        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)

        # lonlat_bbox=hgrid.bbox.bounds
        lonlat_bbox=bbox
        minimum_longitude=lonlat_bbox[0]
        minimum_latitude=lonlat_bbox[1]
        maximum_longitude=lonlat_bbox[0] + lonlat_bbox[2]
        maximum_latitude=lonlat_bbox[1] + lonlat_bbox[3]

        # --- Subset dataset
        output_tmp_filename = outdir / f"{self.dataset_id}_subset_tmp.nc"
        copernicusmarine.subset(
            dataset_id=self.dataset_id,
            variables=variables.keys(),
            minimum_longitude=minimum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_longitude=maximum_longitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_filename = output_tmp_filename, # if extension is .zarr, file is downloaded in Zarr format 
            # output_directory = "./-" # default is current directory
            )

        # --- Reformat Dataset

        ds = xr.open_dataset(output_tmp_filename)

        # Rename variables
        ds = ds.rename(name_dict=variables)

        # # Reset time variable to SCHISM param.nml base time; time in seconds since start_datetime
        # time_in_seconds = ds['time'].copy(data=(ds['time'].values - np.datetime64(start_datetime)) / np.timedelta64(1, 's'))
        #
        # time_in_seconds.encoding['units'] = f'seconds since {start_datetime}'
        #
        # # Create a new DataArray with the converted time values.
        # ds['time'] = time_in_seconds
        #
        # print(ds['time'].encoding)
        #
        # # Update the attributes to match desired output.
        # ds['time'].attrs = {
        #     'long_name': f"time_in_seconds_since_{start_datetime}",
        #     'standard_name': "time",
        #     # 'base_date': [values['start_year'], values['start_month'], values['start_day'], values['start_hour'], 0],
        #     'units': f"seconds since {start_datetime}"
        # }

        # Set time to Julian Days since base date of 1990-01-01
        base_date_str = '1990-01-01 00:00:00'
        base_datetime64 = np.datetime64(base_date_str) # if this throws an error, try using: np.datetime64(start_datetime)

        time_in_julian_days = (ds['time'].values - base_datetime64) / np.timedelta64(1, 'D')
        ds['time'] = time_in_julian_days
        ds['time'].attrs = {
                    'long_name': f"julian day (UT)",
                    'standard_name': "time",
                    # 'base_date': [values['start_year'], values['start_month'], values['start_day'], values['start_hour'], 0],
                    'calendar':"standard",
                    'units': f"days since {base_date_str}",
                    'conventions' : f'relative julian days with decimal part (as parts of the day)',
                    'axis':'T',
                }
        
        ds['longitude'].attrs['units'] = 'degree_east'
        ds['longitude'].attrs['valid_min'] = -180.
        ds['longitude'].attrs['valid_max'] = 360.
        ds['longitude'].attrs['axis'] = 'X'

        ds['latitude'].attrs['units'] = 'degree_north'
        ds['latitude'].attrs['valid_min'] = -90.
        ds['latitude'].attrs['valid_max'] = 180.
        ds['latitude'].attrs['axis'] = 'Y'        

        # hs attr
        ds['hs'].attrs["valid_min"] = 0.
        ds['hs'].attrs["valid_max"] = 100.

        # Add a peak wave frequency variable
        ds["fp"] = 1 / ds["tp"]
        ds["fp"].attrs['long_name']='spectral peak wave frequency'
        ds["fp"].attrs['standard_name']='sea_surface_wave_frequency_at_variance_spectral_density_maximum'
        ds["fp"].attrs['units']='s^-1'
        ds['fp'].attrs["valid_min"] = 0.02
        ds['fp'].attrs["valid_max"] = 2.

        # t02 (wave frequency at mean wave period -- this name does not make sense!)
        ds["t02"] = 1 / ds["t02"] # not sure why it has the same name in SCHISM as the mean wave period ?
        ds["t02"].attrs['long_name']='mean wave frequency'
        ds["t02"].attrs['standard_name']='sea_surface_mean_wave_frequency_from_variance_spectral_density_second_frequency_moment'
        ds["t02"].attrs['units']='s^-1'
        ds['t02'].attrs["valid_min"] = 0.02
        ds['t02'].attrs["valid_max"] = 2.

        # dir
        ds['dir'].attrs["valid_min"] = 0.
        ds['dir'].attrs["valid_max"] = 360.

        # Add directional spreading (estimated!)
        n, dspr_estimate = get_wave_spr_DNV(ds["tp"], showPlots=False)
        ds["spr"] = dspr_estimate
        ds['spr'].attrs["standard_name"] = 'directional_spreading'
        ds['spr'].attrs["long_name"] = 'Mean directional wave spread dspr'
        ds['spr'].attrs["units"] = 'deg'
        ds['spr'].attrs["description"] = 'directional spreading estimate based on sea_surface_wave_period_at_variance_spectral_density_maximum'
        ds['spr'].attrs["valid_min"] = 0.
        ds['spr'].attrs["valid_max"] = 90.

        # Drop tp (not needed for WWM)
        ds = ds.drop_vars("tp")

        # Write each variable to a separate NetCDF file and record the filenames in wwm_inputs.txt.
        # Adjust the output directory if needed (here, files are written to the current directory).
        if os.path.isfile(outdir / self.filewave):
            os.remove(outdir / self.filewave)

        with open(outdir / self.filewave, "w") as f:
            for var in ['dir','fp','hs','spr','t02']: # order of variables matters (or so people claim?)
                filename = f"ww3_{var}.nc".strip()  # Ensure filename is clean.
                f.write(filename + "\n") 
                print("")
                print(outdir / filename)    
                print(ds[var])   
                print("")

                # extract variable
                ds_var = ds[var]

                # apply valid min and valid max
                valid_min = ds_var.attrs.get("valid_min", -np.inf)  # Default -inf if missing
                valid_max = ds_var.attrs.get("valid_max", np.inf)  # Default +inf if missing          
                ds_var = ds_var.where((ds_var >= valid_min) & (ds_var <= valid_max), np.nan) # Mask values outside valid range

                # transpose to expected dimension order from wwm
                # ds_var = ds_var.transpose("longitude", "latitude", "time")
                ds_var = ds_var.transpose("time", "latitude","longitude") # this is the format given with examples of: ncdump -h ww3_hs.nc 

                # set encoding for WWM
                ds_var.encoding["dtype"]="float64"
                ds_var.encoding["_FillValue"]=-9999.0
                ds_var.encoding["scale_factor"]=1.0

                # compression
                ds_var.encoding["zlib"]=True
                ds_var.encoding["complevel"]=4

                # Define encoding for coordinates
                encoding = {
                    "longitude": {"dtype": "float64",'scale_factor': 1.},
                    "latitude": {"dtype": "float64",'scale_factor': 1.},
                    "time": {"dtype": "float64",'scale_factor': 1.},
                    var: ds_var.encoding  # Include variable-specific encoding
                }

                print(ds_var.encoding)

                # write to netcdf
                ds_var.to_netcdf(
                    outdir / filename,
                    unlimited_dims=['time'],
                    encoding=encoding,  # Ensures time, lon, lat are double
                    )

            # Close datasets.
            ds.close()

            # Delete subset dataset 
            subprocess.run(['rm','-rf',f'{output_tmp_filename}'])
            