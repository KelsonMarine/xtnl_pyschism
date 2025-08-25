import xarray as xr
import copernicusmarine
import pyschism.mesh.hgrid 
import wavespectra
import os, pathlib 
import subprocess
from datetime import timedelta, datetime, timezone
import numpy as np
import pandas as pd
import shapely

from pyschism.forcing.wwm.base import WWM, WWM_IBOUNDFORMAT_3
import pyschism.forcing.wwm as wwm 
from typing import Union, Literal
class GLOBAL_MULTIYEAR_WAV_001_032(WWM_IBOUNDFORMAT_3):

    """
    Subset Copernicus Marine Wave Reanalysis Data: GLOBAL_MULTIYEAR_WAV_001_032
    https://doi.org/10.48670/moi-00022
    """

    def __init__(
        self,
        ds: xr.Dataset = None,
        username: str = None,
        password: str = None,
        filepath: Union[pathlib.Path,str]  = None,
        iboundformat = 3 
    ):
    
        """Loads WWM_IBOUNDFORMAT_3 to use as waves input."""
        self.dataset_id = ['cmems_mod_glo_wav_my_0.2deg_PT3H-i', 'cmems_mod_glo_wav_myint_0.2deg_PT3H']

        self.filewave = 'bndfiles.dat'
        # self.WWM_IBOUNDFORMAT_3 = WWM_IBOUNDFORMAT_3(ds=ds)
        self.ds = ds
        self.filepath = filepath
        self.username = username
        self.password = password
        self.iboundformat = iboundformat
        self.outdir = pathlib.Path('./')

    @property
    def variables(self):
        if self.iboundformat==3:
            variables = {
                "VHM0": "hs",   # Significant wave height [m]
                "VTPK": "tp",   # Peak wave period [s]
                "VTM02": "t02", # Mean wave period (second moment) [s]
                "VMDR": "dir",  # Wave direction [degrees]
            }
        elif self.iboundformat==6:
            variables = {
                "VHM0_SW1": "hs_swell_1", # "sea_surface_primary_swell_wave_significant_height",
                "VHM0_SW2": "hs_swell_2", # "sea_surface_secondary_swell_wave_significant_height",
                "VHM0_WW": "hs_windwave", #"sea_surface_wind_wave_significant_height",
                "VMDR_SW1": "dir_swell_1", #"sea_surface_primary_swell_wave_from_direction",
                "VMDR_SW2": "dir_swell_2", #"sea_surface_secondary_swell_wave_from_direction",
                "VMDR_WW": "dir_windwave", # "sea_surface_wind_wave_from_direction",
                "VTM01_SW1": "t01_swell_1", # "sea_surface_primary_swell_wave_mean_period",
                "VTM01_SW2": "t01_swell_2", #"sea_surface_secondary_swell_wave_mean_period",
                "VTM01_WW": "t01_windwave", #"sea_surface_wind_wave_mean_period",
            }
        return variables
    
    # @property
    # def copernicusmarine_subset_filename(self):
    #     # lonlat_bbox=hgrid.bbox.bounds
    #     min_lon=self.bbox[0]
    #     min_lat=self.bbox[1]
    #     max_lon=self.bbox[0] + self.bbox[2]
    #     max_lat=self.bbox[1] + self.bbox[3]
    #     copernicusmarine_subset_filename = self.outdir / f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={min_lon}_{max_lon}_N={min_lat}_{max_lat}_time={self.start_datetime}_{self.end_datetime}.nc"
    #     return copernicusmarine_subset_filename
  
    def describe(self):
        #'GLOBAL_MULTIYEAR_WAV_001_032'
        # dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i' # 15/01/1980 to 30/04/2023 ... likely to change
        # dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H'  # interim period, 01/05/2023 to present (month - 1)
        dataset_describe = copernicusmarine.describe(contains=[self.dataset_id[0]])
        description=dataset_describe.model_dump()
        return description
    
    def download_GLOBAL_MULTIYEAR_WAV_001_032(
            self,
            outdir: Union[str, os.PathLike],
            start_datetime: datetime = None,
            rnday: Union[float, int, timedelta] = None,
            end_datetime: datetime = None,
            bbox: Union[list, tuple] = None,
            hgrid: pyschism.mesh.Hgrid = None, # for use when self.iboundformat == 6 --> use open boundary to define stations
            stations = shapely.LineString,  # for use when self.iboundformat == 6 --> format like [lon, lat] ... then download iteratively to build netcdf
            overwrite: bool = True, # TODO: add error catching here (always overwrites!)
            # cleanup: bool = True
            ) -> pathlib.Path:
        
        if end_datetime is None:
            if not isinstance(rnday,timedelta):
                rnday = timedelta(days=rnday)
            end_datetime = start_datetime + rnday
            
        # define a dict for later renaming
        variables  = self.variables

        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True)

        if self.iboundformat == 3:
            # lonlat_bbox=hgrid.bbox.bounds
            # bbox_bounds = bbox.bounds
            min_lon=bbox.xmin
            min_lat=bbox.ymin
            max_lon=bbox.xmax
            max_lat=bbox.ymax
            
            assert isinstance(outdir, pathlib.Path), "outdir must be a pathlib.Path object"
            filename = (
                    f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={min_lon:1.2f}_{max_lon:1.2f}"
                    f"_N={min_lat:1.2f}_{max_lat:1.2f}"
                    f"_time={start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}.nc"
                )
            copernicusmarine_subset_filename = outdir / filename        
            if not copernicusmarine_subset_filename.is_file():
            # --- Subset dataset
            # copernicusmarine.login(username=self.username,password=self.password,force_overwrite=True)
            # copernicusmarine.subset(
            #         dataset_id='cmems_mod_glo_wav_my_0.2deg_PT3H-i',
            #         variables=variables.keys(),
            #         minimum_longitude=min_lon,
            #         minimum_latitude=min_lat,
            #         maximum_longitude=max_lon,
            #         maximum_latitude=max_lat,
            #         coordinates_selection_method='inside',
            #         start_datetime=start_datetime, 
            #         end_datetime=end_datetime,
            #         output_filename = copernicusmarine_subset_filename, # if extension is .zarr, file is downloaded in Zarr format 
            #         overwrite=overwrite
            #         # output_directory = "./-" # default is current directory
            #         )
                try:
                    filename = (
                        f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={min_lon:1.2f}_{max_lon:1.2f}"
                        f"_N={min_lat:1.2f}_{max_lat:1.2f}"
                        f"_time={start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}.nc"
                    )
                    copernicusmarine_subset_filename = outdir / filename        
                    if not copernicusmarine_subset_filename.is_file():
                        # --- Subset dataset
                        copernicusmarine.login(username=self.username,password=self.password,force_overwrite=True)
                        copernicusmarine.subset(
                            dataset_id='cmems_mod_glo_wav_my_0.2deg_PT3H-i',
                            variables=variables.keys(),
                            minimum_longitude=min_lon,
                            minimum_latitude=min_lat,
                            maximum_longitude=max_lon,
                            maximum_latitude=max_lat,
                            coordinates_selection_method='inside',
                            start_datetime=start_datetime, 
                            end_datetime=end_datetime,
                            output_filename = copernicusmarine_subset_filename, # if extension is .zarr, file is downloaded in Zarr format 
                            overwrite=overwrite
                            # output_directory = "./-" # default is current directory
                            )
                except: 
                    print('subsetting iterim product ... ')
                    filename = (
                        f"cmems_mod_glo_wav_myint_0.2deg_PT3H-i_subset_E={min_lon:1.2f}_{max_lon:1.2f}"
                        f"_N={min_lat:1.2f}_{max_lat:1.2f}"
                        f"_time={start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}.nc"
                    )
                    copernicusmarine_subset_filename = outdir / filename        
                    if not copernicusmarine_subset_filename.is_file():
                        # --- Subset dataset
                        copernicusmarine.login(username=self.username,password=self.password,force_overwrite=True)
                        copernicusmarine.subset(
                            dataset_id='cmems_mod_glo_wav_myint_0.2deg_PT3H-i',
                            variables=variables.keys(),
                            minimum_longitude=min_lon,
                            minimum_latitude=min_lat,
                            maximum_longitude=max_lon,
                            maximum_latitude=max_lat,
                            coordinates_selection_method='inside',
                            start_datetime=start_datetime, 
                            end_datetime=end_datetime,
                            output_filename = copernicusmarine_subset_filename, # if extension is .zarr, file is downloaded in Zarr format 
                            overwrite=overwrite
                            # output_directory = "./-" # default is current directory
                            )                
        elif self.iboundformat == 6:

            raise('not implemented yet')

            if stations:

                if type(stations) == pd.DataFrame:
                    lons = stations.lon.values
                    lats = stations.lat.values
                elif type(stations) == np.ndarray:
                    lons = stations[0,:]
                    lats = stations[1,:]
                elif type(stations) == shapely.LineString:
                    lons = stations.xy[0]
                    lats = stations.xy[1]

            elif hgrid:

                open_boundary_gdf = hgrid.boundaries.open
                lons = open_boundary_gdf.geometry.x
                lats = open_boundary_gdf.geometry.y

            elif bbox:
                raise("'bbox' not implemented when 'iboundformat' == 6 -- download data along open boundary using 'stations' or 'hgrid' as input")

            min_lon = np.min(lons)
            min_lat = np.min(lats)
            max_lon = np.max(lons)
            max_lat = np.max(lats)

            self.lon = lons
            self.lat = lats

            filename = (
            f"cmems_mod_glo_wav_my_0.2deg_PT3H-i_subset_E={min_lon:1.2f}_{max_lon:1.2f}"
            f"_N={min_lat:1.2f}_{max_lat:1.2f}"
            f"_time={start_datetime.strftime('%Y%m%d')}_{end_datetime.strftime('%Y%m%d')}.nc"
            )
            copernicusmarine_subset_filename = outdir / filename        

            if not copernicusmarine_subset_filename.is_file():

                # --- Subset dataset
                copernicusmarine.login(username=self.username,password=self.password,force_overwrite=True)
                # os.makedirs( outdir / 'tmp' ,exist_ok=True)
                # fnames = []
                # for (lon,lat) in zip(lons,lats):
                    # fname = outdir / 'tmp' / f"{self.dataset_id}_{lon}_{lat}.nc"
                copernicusmarine.subset(
                    dataset_id='cmems_mod_glo_wav_my_0.2deg_PT3H-i',
                    variables=variables.keys(),
                    minimum_longitude=min_lon,
                    minimum_latitude=min_lat,
                    maximum_longitude=max_lon,
                    maximum_latitude=max_lat,
                    coordinates_selection_method='inside',
                    start_datetime=start_datetime, 
                    end_datetime=end_datetime,
                    output_filename = copernicusmarine_subset_filename, # if extension is .zarr, file is downloaded in Zarr format 
                    overwrite=overwrite
                    # output_directory = "./-" # default is current directory
                    )
            
        return copernicusmarine_subset_filename

    def format_GLOBAL_MULTIYEAR_WAV_001_032(self,
                                            fname:pathlib.Path,
                                            start_datetime: datetime,
                                            end_datetime: datetime,
                                            bbox,
                                            bbox_buffer = 0.5
                                            ):
        
        # --- Reformat Dataset
        if isinstance(fname, (list, os.PathLike)):
            ds = xr.open_mfdataset(fname)
        else:
            ds = xr.open_dataset(fname)

        if start_datetime is not None and end_datetime is not None:        
            try:
                ds = ds.sel(time=slice(start_datetime, end_datetime))
            except:
                print('assuming tzinfo=None')
                start_datetime = start_datetime.replace(tzinfo=None) # could also be timezone.utc
                end_datetime = end_datetime.replace(tzinfo=None)
                ds = ds.sel(time=slice(start_datetime, end_datetime))

        if bbox is not None:
            if isinstance(bbox,list):
                ds = ds.sel(
                    longitude=slice(bbox[0]-bbox_buffer,bbox[2]+bbox_buffer),
                    latitude=slice(bbox[1]-bbox_buffer,bbox[3]+bbox_buffer)
                    )
            else:
                ds = ds.sel(
                    longitude= slice(bbox.xmin-bbox_buffer,bbox.xmax+bbox_buffer),
                    latitude=slice(bbox.ymin-bbox_buffer,bbox.ymax+bbox_buffer)
                    )


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
        
        if self.iboundformat == 3:

            # Rename variables
            ds = ds.rename(name_dict=self.variables)

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

            self.ds = ds

        elif self.iboundformat == 6:

            raise('not implemented yet')

        # netcdf DUCK94_wave_spectra_8m_array {
        # dimensions:
        #         station = 1 ;
        #         time = 97 ;
        #         frequency = 62 ;
        #         direction = 72 ;
        # variables:
        # double time(time) ;
        #         time:long_name = "julian day (UT)" ;
        #         time:standard_name = "time" ;
        #         time:units = "days since 1990-01-01 00:00:00" ;
        #         time:conventions = "relative julian days with decimal part (as parts of the day )" ;
        # double longitude(time, station) ;
        #         longitude:long_name = "longitude" ;
        #         longitude:standard_name = "longitude" ;
        #         longitude:units = "degree_east" ;
        # double latitude(time, station) ;
        #         latitude:long_name = "latitude" ;
        #         latitude:standard_name = "latitude" ;
        #         latitude:units = "degree_east" ;
        # float frequency(frequency) ;
        #         frequency:long_name = "frequency of center band" ;
        #         frequency:standard_name = "sea_surface_wave_frequency" ;
        #         frequency:globwave_name = "frequency" ;
        #         frequency:units = "s-1" ;
        # float direction(direction) ;
        #         direction:long_name = "sea surface wave to direction" ;
        #         direction:standard_name = "sea_surface_wave_to_direction" ;
        #         direction:globwave_name = "direction" ;
        #         direction:units = "degree" ;
        #         direction:conventions = "0� is True East" ;
        # float efth(time, station, frequency, direction) ;
        #         efth:long_name = "sea surface wave directional variance spectral density" ;
        #         efth:standard_name = "sea_surface_wave_directional_variance_spectral_density" ;
        #         efth:globwave_name = "directional_variance_spectral_density" ;
        #         efth:units = "m2 s rad-1" ;

            self.Directional_Spectra_Wave_Dataset = Directional_Spectra_Wave_Dataset(ds=ds)

            
    def write(self, 
            outdir: pathlib.Path = pathlib.Path('./'), 
            start_datetime: datetime = None,
            rnday: Union[float, int, timedelta] = None,
            end_datetime: datetime = None,
            overwrite: bool = True, # TODO: add error catching here (always overwrites!)
            bbox = None,
            bbox_buffer = 0.5
            ):
        
        if start_datetime is None:
                start_datetime = self.ds.time.isel(time=0).value
        elif end_datetime is None and rnday is not None:
            end_datetime = start_datetime + rnday
        elif rnday is None:
            rnday = end_datetime - start_datetime

        if self.filepath is None:
            copernicusmarine_subset_filename = self.download_GLOBAL_MULTIYEAR_WAV_001_032(
                outdir=outdir,
                start_datetime = start_datetime,
                rnday = rnday,
                end_datetime=end_datetime,
                bbox = bbox,
                overwrite = False,
                )
            bbox = None # drop bbox since it was applied above
        else:
            copernicusmarine_subset_filename = self.filepath

        self.format_GLOBAL_MULTIYEAR_WAV_001_032(
            copernicusmarine_subset_filename,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            bbox=bbox,
            bbox_buffer=bbox_buffer
            )

        WWM_IBOUNDFORMAT_3(ds=self.ds).write(
            outdir=outdir,
            overwrite=overwrite,
            )
        
# class GLOBAL_MULTIYEAR_WAV_001_032_Directional(WWM_IBOUNDFORMAT_6):

#     """
#     Subset Copernicus Marine Wave Reanalysis Dataset: GLOBAL_MULTIYEAR_WAV_001_032

#     Construct model directional spectra from partitioned spectral data: 

#         Sea surface primary swell wave significant height :  VHM0_SW1 [m]

#         Sea surface secondary swell wave significant height :  VHM0_SW2 [m]

#         Sea surface wind wave significant height :  VHM0_WW [m]

#         Sea surface primary swell wave from direction :  VMDR_SW1 [°]

#         Sea surface secondary swell wave from direction :  VMDR_SW2 [°]

#         Sea surface wind wave from direction :  VMDR_WW [°]

#         Sea surface primary swell wave mean period :  VTM01_SW1 [s]

#         Sea surface secondary swell wave mean period :  VTM01_SW2 [s]

#         Sea surface wind wave mean period :  VTM01_WW [s]

#         Sea surface wave mean period from variance spectral density second frequency moment :  VTM02 [s]

#         Sea surface wave mean period from variance spectral density inverse frequency moment :  VTM10 [s]

#     """

#     def __init__(
#         self,
#         dswd: Directional_Spectra_Wave_Dataset = None,
#         dataset_id: Literal['cmems_mod_glo_wav_my_0.2deg_PT3H-i', 'cmems_mod_glo_wav_myint_0.2deg_PT3H'] =  'cmems_mod_glo_wav_myint_0.2deg_PT3H',
#         username: str = None,
#         password: str = None
#     ):
    
#         """Access GLOBAL_MULTIYEAR_WAV_001_032 data to use as waves input."""
#         self.dataset_id = dataset_id
#         self.filewave = 'bndfiles.dat'
#         if dswd is not None:
#             self.dswd = dswd
#         else:
#             if username is not None and password is not None:            
#                 copernicusmarine.login(username=username,password=password,force_overwrite=True)
           

#     def describe(self):
#         #'GLOBAL_MULTIYEAR_WAV_001_032'
#         # dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i' # 15/01/1980 to 30/04/2023
#         # dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H'  # interim period, 01/05/2023 to present (month - 1)
#         dataset_describe = copernicusmarine.describe(contains=[self.dataset_id])
#         description=dataset_describe.model_dump()
#         return description

#     def write(
#         self,
#         outdir: Union[str, os.PathLike],
#         start_datetime: datetime = None,
#         rnday: Union[float, int, timedelta] = None,
#         end_datetime: datetime = None,
#         bbox: Union[list, tuple] = None,
#         overwrite: bool = True, # TODO: add error catching here (always overwrites!)
#         # cleanup: bool = True
#     ):
        
#         """Estimate directional sectra from partioned spectral statistics"""
        
#         if end_datetime is None:
#             if not isinstance(rnday,timedelta):
#                 rnday = timedelta(days=rnday)
#             end_datetime = start_datetime + rnday
        
#         # define variables
#         variables = {
#             "VHM0_SW1": "sea_surface_primary_swell_wave_significant_height",
#             "VHM0_SW2": "sea_surface_secondary_swell_wave_significant_height",
#             "VHM0_WW": "sea_surface_wind_wave_significant_height",
#             "VMDR_SW1": "sea_surface_primary_swell_wave_from_direction",
#             "VMDR_SW2": "sea_surface_secondary_swell_wave_from_direction",
#             "VMDR_WW": "sea_surface_wind_wave_from_direction",
#             "VTM01_SW1": "sea_surface_primary_swell_wave_mean_period",
#             "VTM01_SW2": "sea_surface_secondary_swell_wave_mean_period",
#             "VTM01_WW": "sea_surface_wind_wave_mean_period",
#         }

#         outdir = pathlib.Path(outdir)
#         outdir.mkdir(exist_ok=True)

#         # lonlat_bbox=hgrid.bbox.bounds
#         lonlat_bbox=bbox
#         min_lon=lonlat_bbox[0]
#         min_lat=lonlat_bbox[1]
#         max_lon=lonlat_bbox[0] + lonlat_bbox[2]
#         max_lat=lonlat_bbox[1] + lonlat_bbox[3]

#         # --- Subset dataset
#         output_tmp_filename = outdir / f"{self.dataset_id}_subset_tmp.nc"
#         copernicusmarine.subset(
#             dataset_id=self.dataset_id,
#             variables=variables.keys(),
#             min_lon=min_lon,
#             min_lat=min_lat,
#             max_lon=max_lon,
#             max_lat=max_lat,
#             start_datetime=start_datetime,
#             end_datetime=end_datetime,
#             output_filename = output_tmp_filename, # if extension is .zarr, file is downloaded in Zarr format 
#             # output_directory = "./-" # default is current directory
#             )

#         # --- construct etfh estimate

#         ds = xr.open_dataset(output_tmp_filename)

#         self.construct_etfh()

#         print(self.etfh)

#         # --- write NetCDF file 
#         filepath = outdir / self.filewave
#         if filepath.exists():
#             if overwrite:
#                 os.remove(filepath)
#             else:
#                 raise FileExistsError(f"File '{filepath}' already exists and overwrite is False.")

#         # set encoding for WWM
#         for var in ["time","longitude","latitude","dir","freq","etfh"]:
#             self.etfh[var].encoding["dtype"]="float64"
#             self.etfh[var].encoding["_FillValue"]=-9999.0
#             self.etfh[var].encoding["scale_factor"]=1.0
#             self.etfh[var].encoding["zlib"]=True
#             self.etfh[var].encoding["complevel"]=4

#         # Define encoding for coordinates
#         encoding = {
#             "longitude": {"dtype": "float64",'scale_factor': 1.},
#             "latitude": {"dtype": "float64",'scale_factor': 1.},
#             "time": {"dtype": "float64",'scale_factor': 1.},
#         }

#         # write to netcdf
#         self.etfh.to_netcdf(
#             outdir / 'ww3_efth.nc',
#             unlimited_dims=['time'],
#             encoding=encoding,  # Ensures time, lon, lat are double
#             )

#         # with open(outdir / self.filewave, "w") as f:
#             # for var in ['dir','fp','hs','spr','t02']: # order of variables matters (or so people claim?)
#             #     filename = f"ww3_{var}.nc".strip()  # Ensure filename is clean.
#             #     f.write(filename + "\n") 
#             #     print("")
#             #     print(outdir / filename)    
#             #     print(ds[var])   
#             #     print("")

#             #     # extract variable
#             #     ds_var = ds[var]

#             #     # apply valid min and valid max
#             #     valid_min = ds_var.attrs.get("valid_min", -np.inf)  # Default -inf if missing
#             #     valid_max = ds_var.attrs.get("valid_max", np.inf)  # Default +inf if missing          
#             #     ds_var = ds_var.where((ds_var >= valid_min) & (ds_var <= valid_max), np.nan) # Mask values outside valid range

#             #     # transpose to expected dimension order from wwm
#             #     # ds_var = ds_var.transpose("longitude", "latitude", "time")
#             #     ds_var = ds_var.transpose("time", "latitude","longitude") # this is the format given with examples of: ncdump -h ww3_hs.nc 

#             #     # set encoding for WWM
#             #     ds_var.encoding["dtype"]="float64"
#             #     ds_var.encoding["_FillValue"]=-9999.0
#             #     ds_var.encoding["scale_factor"]=1.0

#             #     # compression
#             #     ds_var.encoding["zlib"]=True
#             #     ds_var.encoding["complevel"]=4

#             #     # Define encoding for coordinates
#             #     encoding = {
#             #         "longitude": {"dtype": "float64",'scale_factor': 1.},
#             #         "latitude": {"dtype": "float64",'scale_factor': 1.},
#             #         "time": {"dtype": "float64",'scale_factor': 1.},
#             #         var: ds_var.encoding  # Include variable-specific encoding
#             #     }

#             #     print(ds_var.encoding)

#             #     # write to netcdf
#             #     ds_var.to_netcdf(
#             #         outdir / filename,
#             #         unlimited_dims=['time'],
#             #         encoding=encoding,  # Ensures time, lon, lat are double
#             #         )

#             # # Close datasets.
#             # ds.close()

#             # # Delete subset dataset 
#             # subprocess.run(['rm','-rf',f'{output_tmp_filename}'])

#     def construct_etfh(
#             self, 
#             freq: Union[list,np.array]=np.logspace(0.04,1,num=24), 
#             direction: Union[list,np.array] = np.arange(0,360,10), 
#             gamma:  Union[list,np.array] = None
#             ):
#         """
#         Reconstruct total directional wave spectra from partitioned wave statistics.

#         Parameters:
#             freq : Frequencies in Hz (default: logspace from ~1.096 mHz to 10 Hz)
#             direction : Directions in degrees (default: 0 to 350 every 10°)

#         Returns:
#             ds_etfh : xarray.Dataset with reconstructed total directional spectra
#         """

#         # # "VHM0_SW1": "sea_surface_primary_swell_wave_significant_height",
#         # # "VMDR_SW1": "sea_surface_primary_swell_wave_from_direction",
#         # # "VTM01_SW1": "sea_surface_primary_swell_wave_mean_period",
    
#         # # "VHM0_SW2": "sea_surface_secondary_swell_wave_significant_height",
#         # # "VMDR_SW2": "sea_surface_secondary_swell_wave_from_direction",
#         # # "VTM01_SW2": "sea_surface_secondary_swell_wave_mean_period",

#         # # "VHM0_WW": "sea_surface_wind_wave_significant_height",
#         # # "VMDR_WW": "sea_surface_wind_wave_from_direction",
#         # # "VTM01_WW": "sea_surface_wind_wave_mean_period",


#         efth_list = []

#         for suffix in ['SW1', 'SW2', 'WW']:
#             hs_key = f'VHM0_{suffix}'
#             dm_key = f'VMDR_{suffix}'
#             ta_key = f'VTM01_{suffix}'

#             if all(k in self.ds for k in [hs_key, dm_key, ta_key]):

#                 ta = self.ds[ta_key].values
#                 hs = self.ds[hs_key].values
#                 dm = self.ds[dm_key].values

#                 if gamma is None:
#                     tp, gamma = wwm.base.solve_Tp_gamma_from_T1_Hs_DNV(Tm01=ta,Hs=hs)
#                 else:
#                     tp = wwm.base.get_wave_Tp_from_Ta_DNV(waveTa=ta, gamma=gamma)

#                 dspr = wwm.base.get_wave_spr_DNV(waveTp=tp)
#                 gth = wavespectra.construct.cartwright(dir=direction, dm=dm, dspr=dspr, under_90=True)
#                 ef = wavespectra.construct.jonswap(freq=freq, fp=1 / tp, gamma=gamma, hs=hs)
#                 efth = ef * gth
#                 efth = efth.fillna(0.0)

#         # Stack all partitions along a new 'partition' dimension
#         efth_concat = xr.concat(efth_list, dim="partition")

#         # Sum over partition dimension to get total energy spectrum
#         efth_total = efth_concat.sum(dim="partition")

#         self.efth = efth_total

#         # # estimate unknown parameters
#         # gamma = 3.3
#         # tp = get_wave_Tp_from_Ta_DNV(waveTa=self.ds['VTM01_SW1'].values, gamma = gamma) 
#         # dspr = get_wave_spr_DNV(waveTp = tp)
#         # gth = wavespectra.construct.direction.cartwright(dir=dir, dm=self.ds['VMDR_SW1'].values, dspr=dspr, under_90=True)
#         # ef = wavespectra.construct.jonswap(freq=freq, fp=1/tp, gamma=gamma, hs= self.ds.VHM0_SW1.values)
#         # efth1 = ef * gth
#         # efth1.fillna(0.0)

    
#     @property
#     def _dir(self):
#         """Wave direction coordinate in coming-from convention, degree"""
#         return self.etfh.dir.values

#     @property
#     def _freq(self):
#         """Wave frequency coordinate, Hz"""
#         return self.etfh.freq.values
    
#     @property
#     def _etfh(self):
#         "Wave 2D spectra, m^2 / degree s"
#         return self.etfh.etfh.values