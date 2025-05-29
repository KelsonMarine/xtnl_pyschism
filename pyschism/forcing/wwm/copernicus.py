import xarray as xr
import copernicusmarine
import pyschism.mesh.hgrid 
import wavespectra
import os, pathlib 
import subprocess
from datetime import timedelta, datetime, timezone
import numpy as np
from pyschism.forcing.wwm.base import WWM, Parametric_Wave_Dataset, Directional_Spectra_Wave_Dataset
import pyschism.forcing.wwm as wwm 
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
        password: str = None,
        filepath: Union[pathlib.Path,str]  = None,
    ):
    
        """Loads Parametric_Wave_Dataset to use as waves input."""
        self.dataset_id = dataset_id
        self.filewave = 'bndfiles.dat'
        if ds is not None:
            self.ds = ds
        elif filepath is not None:
            self.filepath = filepath

        self.username = username
        self.password = password
        self.outdir = pathlib.Path('./')

    @property
    def copernicus_wwm_variable_dict(self):
        return {
            "VHM0": "hs",   # Significant wave height [m]
            "VTPK": "tp",   # Peak wave period [s]
            "VTM02": "t02", # Mean wave period (second moment) [s]
            "VMDR": "dir",  # Wave direction [degrees]
        }
    
    # @property
    # def copernicusmarine_subset_filename(self):
    #     # lonlat_bbox=hgrid.bbox.bounds
    #     minimum_longitude=self.bbox[0]
    #     minimum_latitude=self.bbox[1]
    #     maximum_longitude=self.bbox[0] + self.bbox[2]
    #     maximum_latitude=self.bbox[1] + self.bbox[3]
    #     copernicusmarine_subset_filename = self.outdir / f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={minimum_longitude}_{maximum_longitude}_N={minimum_latitude}_{maximum_latitude}_time={self.start_datetime}_{self.end_datetime}.nc"
    #     return copernicusmarine_subset_filename
  
    def describe(self):
        #'GLOBAL_MULTIYEAR_WAV_001_032'
        # dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i' # 15/01/1980 to 30/04/2023
        # dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H'  # interim period, 01/05/2023 to present (month - 1)
        dataset_describe = copernicusmarine.describe(contains=[self.dataset_id])
        description=dataset_describe.model_dump()
        return description
    
    def download_GLOBAL_MULTIYEAR_WAV_001_032(
            self,
            outdir: Union[str, os.PathLike],
            start_datetime: datetime = None,
            rnday: Union[float, int, timedelta] = None,
            end_datetime: datetime = None,
            bbox: Union[list, tuple] = None,
            overwrite: bool = True, # TODO: add error catching here (always overwrites!)
            # cleanup: bool = True
            ) -> pathlib.Path:
        
        if end_datetime is None:
            if not isinstance(rnday,timedelta):
                rnday = timedelta(days=rnday)
            end_datetime = start_datetime + rnday
            
        # define a dict for later renaming
        variables  = self.copernicus_wwm_variable_dict

        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)

        # lonlat_bbox=hgrid.bbox.bounds
        # bbox_bounds = bbox.bounds
        minimum_longitude=bbox.xmin
        minimum_latitude=bbox.ymin
        maximum_longitude=bbox.xmax
        maximum_latitude=bbox.ymax
        
        assert isinstance(outdir, pathlib.Path), "outdir must be a pathlib.Path object"

        filename = (
            f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={minimum_longitude:1.2f}_{maximum_longitude:1.2f}"
            f"_N={minimum_latitude:1.2f}_{maximum_latitude:1.2f}"
            f"_time={start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}.nc"
        )
        copernicusmarine_subset_filename = outdir / filename        
        if not copernicusmarine_subset_filename.is_file():
            # --- Subset dataset
            copernicusmarine.login(username=self.username,password=self.password,force_overwrite=True)
            copernicusmarine.subset(
                dataset_id=self.dataset_id,
                variables=variables.keys(),
                minimum_longitude=minimum_longitude,
                minimum_latitude=minimum_latitude,
                maximum_longitude=maximum_longitude,
                maximum_latitude=maximum_latitude,
                start_datetime=start_datetime, #-timedelta(hours=6),
                end_datetime=end_datetime,
                output_filename = copernicusmarine_subset_filename, # if extension is .zarr, file is downloaded in Zarr format 
                overwrite=overwrite
                # output_directory = "./-" # default is current directory
                )
        
        return copernicusmarine_subset_filename

    def format_GLOBAL_MULTIYEAR_WAV_001_032(self,fname:pathlib.Path):
        
        # --- Reformat Dataset

        ds = xr.open_dataset(fname)

        # Rename variables
        ds = ds.rename(name_dict=self.copernicus_wwm_variable_dict)

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

        # !! Fix Me: unclear on def of t02 in schism-wwm ... is it mean freq or mean period?

        # # # t02 (mean wave period) 
        ds['t02'] = ds['t02'] + np.nan # test if this var even matters!
        ds['t02'].attrs["valid_min"] = 0
        ds['t02'].attrs["valid_max"] = 50.

        # # t02 (wave frequency at mean wave period -- this name does not make sense!)
        # ds["t02"] = 1 / ds["t02"] # not sure why it has the same name in SCHISM as the mean wave period ?
        # ds["t02"].attrs['long_name']='mean wave frequency'
        # ds["t02"].attrs['standard_name']='sea_surface_mean_wave_frequency_from_variance_spectral_density_second_frequency_moment'
        # ds["t02"].attrs['units']='s^-1'
        # ds['t02'].attrs["valid_min"] = 0.02
        # ds['t02'].attrs["valid_max"] = 2.

        # dir
        ds['dir'].attrs["valid_min"] = 0.
        ds['dir'].attrs["valid_max"] = 360.

        # Add directional spreading (estimated!)
        n, dspr_estimate = wwm.base.get_wave_spr_DNV(ds["tp"], showPlots=False)
        ds["spr"] = dspr_estimate
        ds['spr'].attrs["standard_name"] = 'directional_spreading'
        ds['spr'].attrs["long_name"] = 'Mean directional wave spread dspr'
        ds['spr'].attrs["units"] = 'deg'
        ds['spr'].attrs["description"] = 'directional spreading estimate based on sea_surface_wave_period_at_variance_spectral_density_maximum'
        ds['spr'].attrs["valid_min"] = 0.
        ds['spr'].attrs["valid_max"] = 90.

        # Drop tp (not needed for WWM)
        ds = ds.drop_vars("tp")

        self.pwd = Parametric_Wave_Dataset(ds=ds)
        
    def write(self, 
            outdir: pathlib.Path = pathlib.Path('./'), 
            start_datetime: datetime = None,
            rnday: Union[float, int, timedelta] = None,
            end_datetime: datetime = None,
            overwrite: bool = True, # TODO: add error catching here (always overwrites!)
            bbox = None
            ):
        
        copernicusmarine_subset_filename = self.download_GLOBAL_MULTIYEAR_WAV_001_032(
            outdir=outdir,
            start_datetime = start_datetime,
            rnday = rnday,
            end_datetime=end_datetime,
            bbox = bbox,
            overwrite = False,
            )
        
        self.format_GLOBAL_MULTIYEAR_WAV_001_032(copernicusmarine_subset_filename)

        self.pwd.write(path=outdir,overwrite=overwrite)
        
class GLOBAL_MULTIYEAR_WAV_001_032_Directional(Directional_Spectra_Wave_Dataset):

    """
    Subset Copernicus Marine Wave Reanalysis Dataset: GLOBAL_MULTIYEAR_WAV_001_032

    Construct model directional spectra from partitioned spectral data: 

        Sea surface primary swell wave significant height :  VHM0_SW1 [m]

        Sea surface secondary swell wave significant height :  VHM0_SW2 [m]

        Sea surface wind wave significant height :  VHM0_WW [m]

        Sea surface primary swell wave from direction :  VMDR_SW1 [°]

        Sea surface secondary swell wave from direction :  VMDR_SW2 [°]

        Sea surface wind wave from direction :  VMDR_WW [°]

        Sea surface primary swell wave mean period :  VTM01_SW1 [s]

        Sea surface secondary swell wave mean period :  VTM01_SW2 [s]

        Sea surface wind wave mean period :  VTM01_WW [s]

        Sea surface wave mean period from variance spectral density second frequency moment :  VTM02 [s]

        Sea surface wave mean period from variance spectral density inverse frequency moment :  VTM10 [s]

    """

    def __init__(
        self,
        ds: Directional_Spectra_Wave_Dataset = None,
        dataset_id: Literal['cmems_mod_glo_wav_my_0.2deg_PT3H-i', 'cmems_mod_glo_wav_myint_0.2deg_PT3H'] =  'cmems_mod_glo_wav_myint_0.2deg_PT3H',
        username: str = None,
        password: str = None
    ):
    
        """Access GLOBAL_MULTIYEAR_WAV_001_032 data to use as waves input."""
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
        overwrite: bool = True, # TODO: add error catching here (always overwrites!)
        # cleanup: bool = True
    ):
        
        """Estimate directional sectra from partioned spectral statistics"""
        
        if end_datetime is None:
            if not isinstance(rnday,timedelta):
                rnday = timedelta(days=rnday)
            end_datetime = start_datetime + rnday
        
        # define variables
        variables = {
            "VHM0_SW1": "sea_surface_primary_swell_wave_significant_height",
            "VHM0_SW2": "sea_surface_secondary_swell_wave_significant_height",
            "VHM0_WW": "sea_surface_wind_wave_significant_height",
            "VMDR_SW1": "sea_surface_primary_swell_wave_from_direction",
            "VMDR_SW2": "sea_surface_secondary_swell_wave_from_direction",
            "VMDR_WW": "sea_surface_wind_wave_from_direction",
            "VTM01_SW1": "sea_surface_primary_swell_wave_mean_period",
            "VTM01_SW2": "sea_surface_secondary_swell_wave_mean_period",
            "VTM01_WW": "sea_surface_wind_wave_mean_period",
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

        # --- construct etfh estimate

        ds = xr.open_dataset(output_tmp_filename)

        self.construct_etfh()

        print(self.etfh)

        # --- write NetCDF file 
        filepath = outdir / self.filewave
        if filepath.exists():
            if overwrite:
                os.remove(filepath)
            else:
                raise FileExistsError(f"File '{filepath}' already exists and overwrite is False.")

        # set encoding for WWM
        for var in ["time","longitude","latitude","dir","freq","etfh"]:
            self.etfh[var].encoding["dtype"]="float64"
            self.etfh[var].encoding["_FillValue"]=-9999.0
            self.etfh[var].encoding["scale_factor"]=1.0
            self.etfh[var].encoding["zlib"]=True
            self.etfh[var].encoding["complevel"]=4

        # Define encoding for coordinates
        encoding = {
            "longitude": {"dtype": "float64",'scale_factor': 1.},
            "latitude": {"dtype": "float64",'scale_factor': 1.},
            "time": {"dtype": "float64",'scale_factor': 1.},
        }

        # write to netcdf
        self.etfh.to_netcdf(
            outdir / 'ww3_efth.nc',
            unlimited_dims=['time'],
            encoding=encoding,  # Ensures time, lon, lat are double
            )

        # with open(outdir / self.filewave, "w") as f:
            # for var in ['dir','fp','hs','spr','t02']: # order of variables matters (or so people claim?)
            #     filename = f"ww3_{var}.nc".strip()  # Ensure filename is clean.
            #     f.write(filename + "\n") 
            #     print("")
            #     print(outdir / filename)    
            #     print(ds[var])   
            #     print("")

            #     # extract variable
            #     ds_var = ds[var]

            #     # apply valid min and valid max
            #     valid_min = ds_var.attrs.get("valid_min", -np.inf)  # Default -inf if missing
            #     valid_max = ds_var.attrs.get("valid_max", np.inf)  # Default +inf if missing          
            #     ds_var = ds_var.where((ds_var >= valid_min) & (ds_var <= valid_max), np.nan) # Mask values outside valid range

            #     # transpose to expected dimension order from wwm
            #     # ds_var = ds_var.transpose("longitude", "latitude", "time")
            #     ds_var = ds_var.transpose("time", "latitude","longitude") # this is the format given with examples of: ncdump -h ww3_hs.nc 

            #     # set encoding for WWM
            #     ds_var.encoding["dtype"]="float64"
            #     ds_var.encoding["_FillValue"]=-9999.0
            #     ds_var.encoding["scale_factor"]=1.0

            #     # compression
            #     ds_var.encoding["zlib"]=True
            #     ds_var.encoding["complevel"]=4

            #     # Define encoding for coordinates
            #     encoding = {
            #         "longitude": {"dtype": "float64",'scale_factor': 1.},
            #         "latitude": {"dtype": "float64",'scale_factor': 1.},
            #         "time": {"dtype": "float64",'scale_factor': 1.},
            #         var: ds_var.encoding  # Include variable-specific encoding
            #     }

            #     print(ds_var.encoding)

            #     # write to netcdf
            #     ds_var.to_netcdf(
            #         outdir / filename,
            #         unlimited_dims=['time'],
            #         encoding=encoding,  # Ensures time, lon, lat are double
            #         )

            # # Close datasets.
            # ds.close()

            # # Delete subset dataset 
            # subprocess.run(['rm','-rf',f'{output_tmp_filename}'])

    def construct_etfh(
            self, 
            freq: Union[list,np.array]=np.logspace(0.04,1,num=24), 
            direction: Union[list,np.array] = np.arange(0,360,10), 
            gamma:  Union[list,np.array] = None
            ):
        """
        Reconstruct total directional wave spectra from partitioned wave statistics.

        Parameters:
            freq : Frequencies in Hz (default: logspace from ~1.096 mHz to 10 Hz)
            direction : Directions in degrees (default: 0 to 350 every 10°)

        Returns:
            ds_etfh : xarray.Dataset with reconstructed total directional spectra
        """

        # # "VHM0_SW1": "sea_surface_primary_swell_wave_significant_height",
        # # "VMDR_SW1": "sea_surface_primary_swell_wave_from_direction",
        # # "VTM01_SW1": "sea_surface_primary_swell_wave_mean_period",
    
        # # "VHM0_SW2": "sea_surface_secondary_swell_wave_significant_height",
        # # "VMDR_SW2": "sea_surface_secondary_swell_wave_from_direction",
        # # "VTM01_SW2": "sea_surface_secondary_swell_wave_mean_period",

        # # "VHM0_WW": "sea_surface_wind_wave_significant_height",
        # # "VMDR_WW": "sea_surface_wind_wave_from_direction",
        # # "VTM01_WW": "sea_surface_wind_wave_mean_period",


        efth_list = []

        for suffix in ['SW1', 'SW2', 'WW']:
            hs_key = f'VHM0_{suffix}'
            dm_key = f'VMDR_{suffix}'
            ta_key = f'VTM01_{suffix}'

            if all(k in self.ds for k in [hs_key, dm_key, ta_key]):

                ta = self.ds[ta_key].values
                hs = self.ds[hs_key].values
                dm = self.ds[dm_key].values

                if gamma is None:
                    tp, gamma = wwm.base.solve_Tp_gamma_from_T1_Hs_DNV(Tm01=ta,Hs=hs)
                else:
                    tp = wwm.base.get_wave_Tp_from_Ta_DNV(waveTa=ta, gamma=gamma)

                dspr = wwm.base.get_wave_spr_DNV(waveTp=tp)
                gth = wavespectra.construct.cartwright(dir=direction, dm=dm, dspr=dspr, under_90=True)
                ef = wavespectra.construct.jonswap(freq=freq, fp=1 / tp, gamma=gamma, hs=hs)
                efth = ef * gth
                efth = efth.fillna(0.0)

        # Stack all partitions along a new 'partition' dimension
        efth_concat = xr.concat(efth_list, dim="partition")

        # Sum over partition dimension to get total energy spectrum
        efth_total = efth_concat.sum(dim="partition")

        self.efth = efth_total

        # # estimate unknown parameters
        # gamma = 3.3
        # tp = get_wave_Tp_from_Ta_DNV(waveTa=self.ds['VTM01_SW1'].values, gamma = gamma) 
        # dspr = get_wave_spr_DNV(waveTp = tp)
        # gth = wavespectra.construct.direction.cartwright(dir=dir, dm=self.ds['VMDR_SW1'].values, dspr=dspr, under_90=True)
        # ef = wavespectra.construct.jonswap(freq=freq, fp=1/tp, gamma=gamma, hs= self.ds.VHM0_SW1.values)
        # efth1 = ef * gth
        # efth1.fillna(0.0)

    
    @property
    def _dir(self):
        """Wave direction coordinate in coming-from convention, degree"""
        return self.etfh.dir.values

    @property
    def _freq(self):
        """Wave frequency coordinate, Hz"""
        return self.etfh.freq.values
    
    @property
    def _etfh(self):
        "Wave 2D spectra, m^2 / degree s"
        return self.etfh.etfh.values