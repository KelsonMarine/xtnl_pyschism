import os
import sys
from datetime import datetime,timedelta
import logging
import pathlib
import tempfile
import subprocess
import shutil
from typing import Union
from time import time
from tqdm.auto import tqdm
import math
import numpy as np
import scipy as sp
import requests
#from numba import jit, prange
import netCDF4 as nc
from netCDF4 import Dataset, MFDataset
from matplotlib.transforms import Bbox
import seawater as sw
import xarray as xr

from pyschism.mesh.base import Nodes, Elements
from pyschism.mesh.vgrid import Vgrid

logger = logging.getLogger(__name__)

# shared cache (module-level)
HYCOM_COORD_CACHE = {}

def save_netcdf4_dataset(src_ds: Dataset, out_path, *, format="NETCDF4", zlib=True, complevel=4):
    """
    Copy a netCDF4.Dataset (including remote OPeNDAP subset) to a local .nc file.
    """
    out_path = str(out_path)

    # Create output file
    with Dataset(out_path, "w", format=format) as dst:
        # ---- global attributes ----
        for attr in src_ds.ncattrs():
            dst.setncattr(attr, src_ds.getncattr(attr))

        # ---- dimensions ----
        for dname, dim in src_ds.dimensions.items():
            dst.createDimension(dname, (len(dim) if not dim.isunlimited() else None))

        # ---- variables ----
        for vname, var in src_ds.variables.items():
            # Create variable with same dtype/dims
            fill_value = getattr(var, "_FillValue", None)

            create_kwargs = {}
            if fill_value is not None:
                create_kwargs["fill_value"] = fill_value

            # Compression only works for NETCDF4/NETCDF4_CLASSIC
            if format in ("NETCDF4", "NETCDF4_CLASSIC"):
                create_kwargs.update({"zlib": zlib, "complevel": complevel})

            dst_var = dst.createVariable(vname, var.datatype, var.dimensions, **create_kwargs)

            # Copy variable attributes (except _FillValue which is handled on createVariable)
            for attr in var.ncattrs():
                if attr == "_FillValue":
                    continue
                dst_var.setncattr(attr, var.getncattr(attr))

            # Copy data (this is what actually downloads/materializes the subset)
            dst_var[:] = var[:]

    return out_path

def convert_longitude(ds, bbox):
#https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
#Light_B's solution didn't generate the correct result
#Michael's solution works, but it takes significantly longer to write nc file (~30 mins compared with 5 mins)
#TODO: figure out why it takes much longer with the second method
    #lon_attr = ds.coords['lon'].attrs
    if bbox.xmin < 0:
        logger.info(f'Convert HYCOM longitude from [0, 360) to [-180, 180):')
        #ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds['_lon_adjusted'] = xr.where(ds['lon'] > 180, ds['lon'] - 360, ds['lon'])
    elif bbox.xmin > 0:
        logger.info(f'Convert HYCOM longitude from [-180, 180) to [0, 360): ')
        #ds.coords['lon'] = (ds.coords['lon'] + 360) % 360 - 180
        ds['_lon_adjusted'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])

    t0 = time()
    ds = (
        ds.swap_dims({'lon': '_lon_adjusted'})
        .sel(**{'_lon_adjusted': sorted(ds._lon_adjusted)})
        .drop('lon')
    )
    ds = ds.rename({'_lon_adjusted': 'lon'})
    #ds = ds.sortby(ds.lon)
    #ds.coords['lon'].attrs = lon_attr
    logger.info(f'swap dims took {time()-t0} seconds!')

    #make sure it is clipped to the bbox
    ds = ds.sel(lon=slice(bbox.xmin - 0.5, bbox.xmax + 0.5))

    return ds

def get_database(date, Bbox=None):
    if date >= datetime(2018, 12, 4):
        database = f'GLBy0.08/expt_93.0'
    elif date >= datetime(2018, 1, 1) and date < datetime(2018, 12, 4):
        database = f'GLBv0.08/expt_93.0'
    elif date >= datetime(2017, 10, 1) and date < datetime(2018, 1, 1):
        database = f'GLBv0.08/expt_92.9'
    elif date >= datetime(2017, 6, 1) and date < datetime(2017, 10, 1):
        database = f'GLBv0.08/expt_57.7'
    elif date >= datetime(2017, 2, 1) and date < datetime(2017, 6, 1):
        database = f'GLBv0.08/expt_92.8'
    elif date >= datetime(2016, 5, 1) and date < datetime(2017, 2, 1):
        database = f'GLBv0.08/expt_57.2'
    elif date >= datetime(2016, 1, 1) and date < datetime(2016, 5, 1):
        database = f'GLBv0.08/expt_56.3'
    elif date >= datetime(1994, 1, 1) and date < datetime(2016, 1, 1):
        database = f'GLBv0.08/expt_53.X/data/{date.year}'
    else:
        raise ValueError(f'No data fro {date}!')
    return database

def get_idxs(date, database, bbox, lonc=None, latc=None):

    if date >= datetime.utcnow():
        date2 = datetime.utcnow() - timedelta(days=1)
        baseurl = f'https://tds.hycom.org/thredds/dodsC/{database}/FMRC/runs/GLBy0.08_930_FMRC_RUN_{date2.strftime("%Y-%m-%dT12:00:00Z")}?depth[0:1:-1],lat[0:1:-1],lon[0:1:-1],time[0:1:-1]'
    else:
        baseurl=f'https://tds.hycom.org/thredds/dodsC/{database}?lat[0:1:-1],lon[0:1:-1],time[0:1:-1],depth[0:1:-1]'

    ds=Dataset(baseurl)
    lonvar='lon'
    latvar='lat'

    time1=ds['time']
    times=nc.num2date(time1,units=time1.units,only_use_cftime_datetimes=False)

    lon=ds[lonvar][:]
    lat=ds[latvar][:]
    dep=ds['depth'][:]

    #check if hycom's lon is the same range as schism's
    same = True
    if not (bbox.xmin >= lon.min() and bbox.xmax <= lon.max()):
        same = False
        if lon.min() >= 0:
            logger.info(f'Convert HYCOM longitude from [0, 360) to [-180, 180):')
            idxs = lon>=180
            lon[idxs] = lon[idxs]-360
        elif lon.min() <= 0:
            logger.info(f'Convert HYCOM longitude from [-180, 180) to [0, 360):')
            idxs = lon<=0
            lon[idxs] = lon[idxs]+360

    lat_idxs=np.where((lat>=bbox.ymin-0.5)&(lat<=bbox.ymax+0.5))[0]
    lon_idxs=np.where((lon>=bbox.xmin-0.5) & (lon<=bbox.xmax+0.5))[0]
    lon=lon[lon_idxs]
    lat=lat[lat_idxs]
    #logger.info(lon_idxs)
    #logger.info(lat_idxs)
    lon_idx1=lon_idxs[0].item()
    lon_idx2=lon_idxs[-1].item()
    #logger.info(f'lon_idx1 is {lon_idx1}, lon_idx2 is {lon_idx2}')
    lat_idx1=lat_idxs[0].item()
    lat_idx2=lat_idxs[-1].item()
    #logger.info(f'lat_idx1 is {lat_idx1}, lat_idx2 is {lat_idx2}')

    if lonc is None:
        lonc = lon.mean()
    #logger.info(f'lonc is {lonc}')
    if latc is None:
        latc = lat.mean()
    #logger.info(f'latc is {latc}')
    x2, y2=transform_ll_to_cpp(lon, lat, lonc, latc)

    idxs=np.where( date == times)[0]
    #check if time_idx is empty
    if len(idxs) == 0:
        #If there is missing data, use the data from the next days, the maximum searching days is 3. Otherwise, stop.
        for i in np.arange(0,3):
            date_before=(date + timedelta(days=int(i)+1)) #.astype(datetime)
            logger.info(f'Try replacing the missing data from {date_before}')
            idxs=np.where(date_before == times)[0]
            if len(idxs) == 0:
                continue
            else:
                break
    if len(idxs) ==0:
        logger.info(f'No date for date {date}')
        sys.exit()
    time_idx=idxs.item()

    ds.close()

    return time_idx, lon_idx1, lon_idx2, lat_idx1, lat_idx2, x2, y2, same

def _hycom_coord_url_and_key(date, database):
    """
    Returns (baseurl, cache_key) for the coordinate-only dataset.
    """
    if date >= datetime.utcnow():
        date2 = datetime.utcnow() - timedelta(days=1)
        runstamp = date2.strftime("%Y-%m-%dT12:00:00Z")
        baseurl = (
            f'https://tds.hycom.org/thredds/dodsC/{database}/FMRC/runs/'
            f'GLBy0.08_930_FMRC_RUN_{runstamp}'
            f'?depth[0:1:-1],lat[0:1:-1],lon[0:1:-1],time[0:1:-1]'
        )
        cache_key = (database, "fmrc", runstamp)
    else:
        baseurl = (
            f'https://tds.hycom.org/thredds/dodsC/{database}'
            f'?lat[0:1:-1],lon[0:1:-1],time[0:1:-1],depth[0:1:-1]'
        )
        cache_key = (database, "hindcast", None)

    return baseurl, cache_key

def get_idxs_and_time_range(day_start, database, bbox, lonc=None, latc=None, max_shift_days=3, cache=None):
    """
    Open HYCOM coords/time once (cached) and return:
      t_start, t_end (inclusive indices for day_start <= time < day_start+1 day),
      lon_idx1, lon_idx2, lat_idx1, lat_idx2,
      x2, y2, isLonSame,
      shifted_day_start (if missing data and shifted forward).

    cache: dict
      Optional external cache so callers can share state across runs/modules.
      If None, uses module-level HYCOM_COORD_CACHE.
    """
    if cache is None:
        cache = HYCOM_COORD_CACHE

    baseurl, cache_key = _hycom_coord_url_and_key(day_start, database)

    # ---- load coords/time from cache or server ----
    if cache_key not in cache:
        ds = Dataset(baseurl)

        time_var = ds["time"]
        times_py = nc.num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)

        # Convert to numpy datetime64 for fast vector comparisons
        times = np.array(times_py, dtype="datetime64[ns]")

        lon = ds["lon"][:].astype(float)
        lat = ds["lat"][:].astype(float)
        depth = ds["depth"][:]  # not required for idx calc but often useful elsewhere

        ds.close()

        cache[cache_key] = {
            "times": times,
            "lon": lon,
            "lat": lat,
            "depth": depth,
            "baseurl": baseurl,
        }
        logger.info(f"Cached HYCOM coords/time for {cache_key} (nt={times.size}, nlon={lon.size}, nlat={lat.size})")

    entry = cache[cache_key]
    times = entry["times"]
    lon = entry["lon"].copy()  # copy because we may modify lon range
    lat = entry["lat"]         # lat not modified
    # depth = entry["depth"]   # available if needed

    # ---- lon range check / convert like your original get_idxs() ----
    isLonSame = True
    if not (bbox.xmin >= lon.min() and bbox.xmax <= lon.max()):
        isLonSame = False
        if lon.min() >= 0:
            logger.info('Convert HYCOM longitude from [0, 360) to [-180, 180):')
            idxs = lon >= 180
            lon[idxs] = lon[idxs] - 360
        elif lon.min() <= 0:
            logger.info('Convert HYCOM longitude from [-180, 180) to [0, 360):')
            idxs = lon <= 0
            lon[idxs] = lon[idxs] + 360

    # ---- spatial indices ----
    lat_idxs = np.where((lat >= bbox.ymin - 0.5) & (lat <= bbox.ymax + 0.5))[0]
    lon_idxs = np.where((lon >= bbox.xmin - 0.5) & (lon <= bbox.xmax + 0.5))[0]

    if lat_idxs.size == 0 or lon_idxs.size == 0:
        raise ValueError("BBox does not intersect HYCOM lat/lon grid after lon conversion.")

    lon_idx1 = int(lon_idxs[0])
    lon_idx2 = int(lon_idxs[-1])
    lat_idx1 = int(lat_idxs[0])
    lat_idx2 = int(lat_idxs[-1])

    lon_sub = lon[lon_idxs]
    lat_sub = lat[lat_idxs]

    if lonc is None:
        lonc = float(lon_sub.mean())
    if latc is None:
        latc = float(lat_sub.mean())

    x2, y2 = transform_ll_to_cpp(lon_sub, lat_sub, lonc, latc)

    # ---- time range indices for the day ----
    shifted_day_start = day_start
    t_start = None
    t_end = None
    for shift in range(0, max_shift_days + 1):
        ds_day_start = day_start + timedelta(days=shift)
        ds_day_end = ds_day_start + timedelta(days=1)

        start64 = np.datetime64(ds_day_start, "ns")
        end64 = np.datetime64(ds_day_end, "ns")

        idx = np.where((times >= start64) & (times < end64))[0]
        if idx.size > 0 and t_start is None:
            t_start = int(idx[0])
            t_end = int(idx[-1])
            shifted_day_start = ds_day_start

    if t_start is None:
        logger.info(f'No HYCOM time coverage found for {day_start} (or next {max_shift_days} days).')
        return None, None, None, None, None, None, None, None, isLonSame, day_start

    return (t_start, t_end, lon_idx1, lon_idx2, lat_idx1, lat_idx2, x2, y2, isLonSame, shifted_day_start)

def floor_2_decimals(number):
    """
    Rounds a number down to exactly two decimal places.
    """
    return math.floor( number * 100) / 100

def ceil_2_decimals(number):
    """
    Rounds a number up to two decimal places.
    """
    return math.ceil(number * 100) / 100.0

def transform_ll_to_cpp(lon, lat, lonc=-77.07, latc=24.0):
    longitude=lon/180*np.pi
    latitude=lat/180*np.pi
    radius=6378206.4
    loncc=lonc/180*np.pi
    latcc=latc/180*np.pi
    lon_new=[radius*(longitude[i]-loncc)*np.cos(latcc) for i in np.arange(len(longitude))]
    lat_new=[radius*latitude[i] for i in np.arange(len(latitude))]

    return np.array(lon_new), np.array(lat_new)

def interp_to_points_3d(dep, y2, x2, bxyz, val):
    # print(type(val), val.dtype, np.ma.isMaskedArray(val),'val shape',val.shape)
    val = np.asanyarray(val).astype(float)
    idxs = np.where(abs(val) > 10000)
    val[idxs] = float('nan')

    if not np.all(x2[:-1] <= x2[1:]):
        logger.info('x2 is not in stricitly ascending order! Sorting x2 and val')
        idxs = np.argsort(x2)
        x2 = x2[idxs]
        val = val[:, :, idxs]

    val_fd = sp.interpolate.RegularGridInterpolator((dep,y2,x2),np.squeeze(val),'linear', bounds_error=False, fill_value = float('nan'))
    val_int = val_fd(bxyz)
    idxs = np.isnan(val_int)
    isgood = np.all(np.isfinite(bxyz[~idxs,:]), axis=1) & np.isfinite(val_int[~idxs])

    if np.sum(idxs) != 0:
        try:
            # !! FIX ME !!
            # there seems to be an error here ... when bxyz comes from an LSC2 vgrid, there are nan values in the 0 column of bxyz
            val_int[idxs] = sp.interpolate.griddata(bxyz[~idxs,:][isgood], val_int[~idxs][isgood], bxyz[idxs,:],'nearest')
        except:
            print(f'hycom2schism.interp_to_points_3d failed on {np.sum(idxs)} of {idxs.size} points ... using nanmean to fill nans ... \n')
            val_int[idxs] = np.nanmean(val_int,axis=0)

    idxs = np.isnan(val_int)
    if np.sum(idxs) != 0:
        logger.info(f'interp_to_points_3d error! There is still missing value for {val}')
        sys.exit()
    return val_int

def interp_to_points_2d(y2, x2, bxy, val):
    # print(type(val), val.dtype, np.ma.isMaskedArray(val))
    val = np.asanyarray(val).astype(float)
    idxs = np.where(abs(val) > 10000)
    val[idxs] = float('nan')

    if not np.all(x2[:-1] <= x2[1:]):
        logger.info('x2 is not in stricitly ascending order! Sorting x2 and val')
        idxs = np.argsort(x2)
        x2 = x2[idxs]
        val = val[:, idxs]

    val_fd = sp.interpolate.RegularGridInterpolator(
        (y2,x2),
        np.squeeze(val),
        'linear',
        bounds_error=False, 
        fill_value = float('nan')
        )
    val_int = val_fd(bxy)
    idxs = np.isnan(val_int)
    if np.sum(idxs) != 0:
        if not val_int[~idxs].size ==0:
            val_int[idxs] = sp.interpolate.griddata(bxy[~idxs,:], val_int[~idxs], bxy[idxs,:],'nearest')
        else:
            logger.info(f'Filling missing values for {val} as 0')
            val_int[idxs] =0

    idxs = np.isnan(val_int)
    if np.sum(idxs) != 0:
        logger.info(f'There is still missing value for {val}')
        print('hycom2schism.py interp_to_points_2d error! Exiting ... you should modify problemeatic open boundary ...')
        sys.exit()
    return val_int

def ConvertTemp(salt, temp, dep):
    nz = temp.shape[0]
    ny = temp.shape[1]
    nx = temp.shape[2]
    pr = np.ones(temp.shape)
    pre = pr*dep[:,None, None]
    Pr = np.zeros(temp.shape)
    ptemp = sw.ptmp(salt, temp, pre, Pr)*1.00024
    return ptemp

class OpenBoundaryInventory:

    def __init__(self, hgrid, vgrid=None, ocean_bnd_ids: list = [0]):
        self.hgrid = hgrid
        if vgrid is None:
            print('OpenBoundaryInventory using default Vgrid')
            vgrid = Vgrid.default() 
        elif isinstance(vgrid, os.PathLike) or  isinstance(vgrid, str):
            vgrid = Vgrid.open(vgrid)
        self.vgrid = vgrid
        self.ocean_bnd_ids = ocean_bnd_ids

    def fetch_data(self, 
                   outdir: Union[str, os.PathLike], 
                   start_date, 
                   rnday, 
                   elev2D=True, 
                   TS=True, 
                   UV=True, 
                   restart=False, 
                   adjust2D=False, 
                   lats=None, 
                   msl_shifts=None,
                   ocean_bnd_ids = None, # can reset from initalized value
                   overwrite = True,
                   archive_netcdf=False,
                   archivedir = None
                   ): 
        outdir = pathlib.Path(outdir)

        if archive_netcdf:  # create new download dir
            if archivedir is None:
                archivedir = outdir
            archivedir = pathlib.Path(f'{archivedir}/hycom/')
            archivedir.mkdir(exist_ok=True,parents=True)  

        if elev2D and not overwrite and pathlib.Path(outdir / 'elev2D.th.nc').exists():
            elev2D = False
            print(f'OpenBoundaryInventory.fetch_data setting elev2D=False (file exists: {outdir / "elev2D.th.nc"})')
        if UV and not overwrite and pathlib.Path(outdir / 'UV3D.th.nc').exists():
            UV = False
            print(f'OpenBoundaryInventory.fetch_data setting UV=False (file exists: {outdir / "SAL_3D.th.nc"})')
        if TS and not overwrite and pathlib.Path(outdir / 'SAL_3D.th.nc').exists() and pathlib.Path(outdir / 'TEM_3D.th.nc').exists():
            TS = False
            print(f'OpenBoundaryInventory.fetch_data setting TS=False (files exists: {outdir / "SAL_3D.th.nc"} and/or {"TEM_3D.th.nc"})')

        self.start_date = start_date
        if not isinstance(rnday,timedelta):
            rnday = timedelta(days=rnday)
        self.rnday=rnday
        self.timevector=np.arange(
            self.start_date,
            self.start_date + self.rnday + timedelta(days=1),
            timedelta(days=1)).astype(datetime)

        #Get open boundary
        gdf=self.hgrid.boundaries.open.copy()
        opbd=[]
        if ocean_bnd_ids is None:
            ocean_bnd_ids = list(range(gdf.shape[0]))
        for ibnd in ocean_bnd_ids:
            opbd.extend(list(gdf.iloc[ibnd].indexes))
        blon = self.hgrid.coords[opbd,0]
        blat = self.hgrid.coords[opbd,1]
        #logger.info(f'blon min {np.min(blon)}, max {np.max(blon)}')
        NOP = len(blon)

        #calculate zcor for 3D
        if TS or UV:
            sigma=self.vgrid.sigma

            #get bathymetry
            depth = self.hgrid.values

            #compute zcor
            zcor = depth[:,None]*sigma
            nvrt=zcor.shape[1]

            #zcor2=zcor[opbd,:]
            #idxs=np.where(zcor2 > 5000)
            #zcor2[idxs]=5000.0-1.0e-6

            #construct schism grid
            #x2i=np.tile(xi,[nvrt,1]).T
            #y2i=np.tile(yi,[nvrt,1]).T
            #bxyz=np.c_[zcor2.reshape(np.size(zcor2)),y2i.reshape(np.size(y2i)),x2i.reshape(np.size(x2i))]
            #logger.info('Computing SCHISM zcor is done!')

        #create netcdf
        ntimes=self.rnday+timedelta(days=1)
        nComp1=1
        nComp2=2
        one=1
        #ndt=np.zeros([ntimes])

        if elev2D and restart == False:
            #timeseries_el=np.zeros([ntimes,NOP,nComp1])
            #create netcdf
            dst_elev = Dataset(outdir / 'elev2D.th.nc', 'w', format='NETCDF4')
            #dimensions
            dst_elev.createDimension('nOpenBndNodes', NOP)
            dst_elev.createDimension('one', one)
            dst_elev.createDimension('time', None)
            dst_elev.createDimension('nLevels', one)
            dst_elev.createDimension('nComponents', nComp1)

            #variables
            dst_elev.createVariable('time_step', 'f', ('one',))
            dst_elev['time_step'][:] = 86400

            dst_elev.createVariable('time', 'f', ('time',))
            #dst_elev['time'][:] = ndt

            dst_elev.createVariable('time_series', 'f', ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'))
            #dst_elev['time_series'][:,:,:,:] = timeseries_el
        elif elev2D and restart:
            dst_elev = Dataset(outdir / 'elev2D.th.nc', 'a', format='NETCDF4')
            time_idx_restart = dst_elev['time'][:].shape[0]

        if TS and restart == False:
            #timeseries_s=np.zeros([ntimes,NOP,nvrt,nComp1])
            dst_salt = Dataset(outdir / 'SAL_3D.th.nc', 'w', format='NETCDF4')
            #dimensions
            dst_salt.createDimension('nOpenBndNodes', NOP)
            dst_salt.createDimension('one', one)
            dst_salt.createDimension('time', None)
            dst_salt.createDimension('nLevels', nvrt)
            dst_salt.createDimension('nComponents', nComp1)
            #variables
            dst_salt.createVariable('time_step', 'f', ('one',))
            dst_salt['time_step'][:] = 86400

            dst_salt.createVariable('time', 'f', ('time',))
            #dst_salt['time'][:] = ndt

            dst_salt.createVariable('time_series', 'f', ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'))

            #temp
            #timeseries_t=np.zeros([ntimes,NOP,nvrt,nComp1])

            dst_temp = Dataset(outdir / 'TEM_3D.th.nc', 'w', format='NETCDF4')
            #dimensions
            dst_temp.createDimension('nOpenBndNodes', NOP)
            dst_temp.createDimension('one', one)
            dst_temp.createDimension('time', None)
            dst_temp.createDimension('nLevels', nvrt)
            dst_temp.createDimension('nComponents', nComp1)
            #variables
            dst_temp.createVariable('time_step', 'f', ('one',))
            dst_temp['time_step'][:] = 86400

            dst_temp.createVariable('time', 'f', ('time',))
            #dst_temp['time'][:] = ndt

            dst_temp.createVariable('time_series', 'f', ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'))
            #dst_temp['time_series'][:,:,:,:] = timeseries_t
        elif TS and restart:
            dst_salt = Dataset(outdir / 'SAL_3D.th.nc', 'a', format='NETCDF4')
            dst_temp = Dataset(outdir / 'TEM_3D.th.nc', 'a', format='NETCDF4')
            time_idx_restart = dst_salt['time'][:].shape[0]

        if UV and restart == False:
            #timeseries_uv=np.zeros([ntimes,NOP,nvrt,nComp2])
            dst_uv = Dataset(outdir / 'uv3D.th.nc', 'w', format='NETCDF4')
            #dimensions
            dst_uv.createDimension('nOpenBndNodes', NOP)
            dst_uv.createDimension('one', one)
            dst_uv.createDimension('time', None)
            dst_uv.createDimension('nLevels', nvrt)
            dst_uv.createDimension('nComponents', nComp2)
            #variables
            dst_uv.createVariable('time_step', 'f', ('one',))
            dst_uv['time_step'][:] = 86400

            dst_uv.createVariable('time', 'f', ('time',))
            #dst_uv['time'][:] = ndt

            dst_uv.createVariable('time_series', 'f', ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'))
            #dst_uv['time_series'][:,:,:,:] = timeseries_uv

        elif UV and restart:
            dst_uv = Dataset(outdir / 'uv3D.th.nc', 'a', format='NETCDF4')
            time_idx_restart = dst_uv['time'][:].shape[0]

        logger.info('**** Accessing GOFS data *****')
        t0=time()

        if restart == False:
            timevector = self.timevector
            it0 = 0
        elif restart:
            #restart from one day earlier
            timevector = self.timevector[time_idx_restart-1:]
            it0 = time_idx_restart-1

        for it1, date in enumerate(timevector):
            it = it0 + it1                

            database=get_database(date)
            logger.info(f'Data for {date} from database {database}')
            print(f'Data for {date} from database {database}')

            #loop over each open boundary
            ind1 = 0
            ind2 = 0
            #for boundary in gdf.itertuples():
            for ibnd in ocean_bnd_ids:
                #opbd = list(boundary.indexes)
                opbd = list(gdf.iloc[ibnd].indexes)
                ind1 = ind2
                ind2 = ind1 + len(opbd)
                #logger.info(f'ind1 = {ind1}, ind2 = {ind2}')
                blon = self.hgrid.coords[opbd,0]
                blat = self.hgrid.coords[opbd,1]
                blonc = blon.mean()
                blatc = blat.mean()
                #logger.info(f'blonc = {blon.mean()}, blatc = {blat.mean()}')
                xi,yi = transform_ll_to_cpp(blon, blat, blonc, blatc)
                bxy = np.c_[yi, xi]

                if TS or UV:
                    zcor2=zcor[opbd,:]
                    idxs=np.where(zcor2 > 5000)
                    zcor2[idxs]=5000.0-1.0e-6

                    #construct schism grid
                    x2i=np.tile(xi,[nvrt,1]).T
                    y2i=np.tile(yi,[nvrt,1]).T
                    bxyz=np.c_[zcor2.reshape(np.size(zcor2)),y2i.reshape(np.size(y2i)),x2i.reshape(np.size(x2i))]


                xmin = floor_2_decimals(np.min(blon))
                xmax = ceil_2_decimals(np.max(blon))
                ymin= floor_2_decimals(np.min(blat))
                ymax = ceil_2_decimals(np.max(blat))
                bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)
                
                time_idx, lon_idx1, lon_idx2, lat_idx1, lat_idx2, x2, y2, _ = get_idxs(date, database, bbox, lonc=blonc, latc=blatc)
                            
                if date >= datetime.utcnow():
                    date2 = datetime.utcnow() - timedelta(days=1)
                    url = f'https://tds.hycom.org/thredds/dodsC/{database}/FMRC/runs/GLBy0.08_930_FMRC_RUN_' + \
                        f'{date2.strftime("%Y-%m-%dT12:00:00Z")}?depth[0:1:-1],lat[{lat_idx1}:1:{lat_idx2}],' + \
                        f'lon[{lon_idx1}:1:{lon_idx2}],time[{time_idx}],' + \
                        f'surf_el[{time_idx}][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_temp[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'salinity[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_u[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_v[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}]'
                else:
                    url=f'https://tds.hycom.org/thredds/dodsC/{database}?lat[{lat_idx1}:1:{lat_idx2}],' + \
                        f'lon[{lon_idx1}:1:{lon_idx2}],depth[0:1:-1],time[{time_idx}],' + \
                        f'surf_el[{time_idx}][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_temp[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'salinity[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_u[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'water_v[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}]'
                logger.info(url)
         
                # Generate a local copy of the HYCOM file for future use
                if archive_netcdf:
                    hycom_fname = archivedir / f'hycom_{date.strftime("%Y%m%d_%H")}_{xmin:0.3f}_{xmax:0.3f}E__{ymin:0.3f}_{ymax:0.3f}N.nc'
                    if not hycom_fname.exists():
                        logger.info(f'Local copy of HYCOM file {hycom_fname} does not exist.')
                        logger.info(f'Downloading HYCOM data from {database} ... ')
                        xr.open_dataset(url).to_netcdf(hycom_fname)
                        logger.info(f'Data saved to {hycom_fname.resolve()}')
                    try:
                        logger.info(f'Reading HYCOM file: {hycom_fname}')
                        ds=Dataset(hycom_fname)
                    except:
                        logger.info(f'Failed to open local copy of HYCOM file {hycom_fname} ... Downloading...')
                        ds=Dataset(url)
                else:
                    logger.info(f'Downloading HYCOM data from {database} ... ')
                    ds=Dataset(url)

                dep=ds['depth'][:]

                logger.info(f'****Interpolation starts for boundary {ibnd}****')

                #ndt[it]=it*24*3600.

                if elev2D:
                    #ssh
                    ssh=np.squeeze(ds['surf_el'][:,:])

                    ssh_int = interp_to_points_2d(y2, x2, bxy, ssh)
                    dst_elev['time'][it] = it*24*3600.
                    if adjust2D:
                        elev_adjust = np.interp(blat, lats, msl_shifts)
                        dst_elev['time_series'][it,ind1:ind2,0,0] = ssh_int + elev_adjust
                    else:
                        dst_elev['time_series'][it,ind1:ind2,0,0] = ssh_int

                if TS:
                    #salt
                    salt = np.squeeze(ds['salinity'][:,:,:])

                    salt_int = interp_to_points_3d(dep, y2, x2, bxyz, salt)
                    salt_int = salt_int.reshape(zcor2.shape)
                    #timeseries_s[it,:,:,0]=salt_int
                    dst_salt['time'][it] = it*24*3600.
                    dst_salt['time_series'][it,ind1:ind2,:,0] = salt_int

                    #temp
                    temp = np.squeeze(ds['water_temp'][:,:,:])

                    #Convert temp to potential temp
                    ptemp = ConvertTemp(salt, temp, dep)

                    temp_int = interp_to_points_3d(dep, y2, x2, bxyz, ptemp)
                    temp_int = temp_int.reshape(zcor2.shape)
                    #timeseries_t[it,:,:,0]=temp_int
                    dst_temp['time'][it] = it*24*3600.
                    dst_temp['time_series'][it,ind1:ind2,:,0] = temp_int

                if UV:
                    uvel=np.squeeze(ds['water_u'][:,:,:])
                    vvel=np.squeeze(ds['water_v'][:,:,:])

                    dst_uv['time'][it] = it*24*3600.
                    #uvel
                    uvel_int = interp_to_points_3d(dep, y2, x2, bxyz, uvel)
                    uvel_int = uvel_int.reshape(zcor2.shape)
                    dst_uv['time_series'][it,ind1:ind2,:,0] = uvel_int

                    #vvel
                    vvel_int = interp_to_points_3d(dep, y2, x2, bxyz, vvel)
                    vvel_int = vvel_int.reshape(zcor2.shape)
                    dst_uv['time_series'][it,ind1:ind2,:,1] = vvel_int
                    #timeseries_uv[it,:,:,1]=vvel_int

                ds.close()
        logger.info(f'Writing *th.nc takes {time()-t0} seconds')

class Nudge:

    def __init__(self, hgrid=None, ocean_bnd_ids=None):

        if hgrid is None:
            raise ValueError('No hgrid information!')
        else:
            self.hgrid = hgrid


        if ocean_bnd_ids is None:
            raise ValueError('Please specify indexes for ocean boundaries!')
        else:
            self.ocean_bnd_ids = ocean_bnd_ids


    def gen_nudge(self, outdir: Union[str, os.PathLike], rlmax = 1.5, rnu_day=0.25):
        """
        set up nudge zone within rlmax distance from the ocean boundary;
        modify the nudging zone width rlmax.
        rlmax can be a uniform value, e.g., rlmax = 1.5 (degree if hgrid is lon/lat)

        #rlmax - max relax distance in m or degree
        #rnu_day - max relax strength in days 
        #restart = True will append to the existing nc file, works when first try doesn't break.

        """

        outdir = pathlib.Path(outdir)

        rnu_max = 1.0 / rnu_day / 86400.0

        #get nudge zone
        lon = self.hgrid.coords[:,0]
        lat = self.hgrid.coords[:,1]
        gdf = self.hgrid.boundaries.open.copy()
        elnode = self.hgrid.elements.array
        NE, NP = elnode.shape[0],len(lon)
        nudge_coeff = np.zeros(NP, dtype=float)

        global_idxs = {}

        t0 = time()
        nudge_coeff = np.zeros(NP, dtype=float)
        for i in self.ocean_bnd_ids:
            print(f'boundary {i}')
            bnd_idxs = gdf.iloc[i].indexes

            dis = abs((lon + 1j*lat)[:, None] - (lon[bnd_idxs] + 1j*lat[bnd_idxs])[None, :]).min(axis=1)
            out = (1-dis/rlmax)*rnu_max
            out[out<0] = 0
            out[out>rnu_max] = rnu_max
            fp = out>0
            nudge_coeff[fp] = np.maximum(out[fp], nudge_coeff[fp])

            idxs_nudge=np.zeros(NP, dtype=int)
            idxs=np.where(out > 0)[0]
            idxs_nudge[idxs]=1

            #expand nudging marker to neighbor nodes
            i34 = self.hgrid.elements.i34
            fp = i34==3
            idxs=np.where(np.max(out[elnode[fp, 0:3]], axis=1) > 0)[0]
            idxs_nudge[elnode[fp,0:3][idxs,:]]=1
            idxs=np.where(np.max(out[elnode[~fp, :]], axis=1) > 0)[0]
            idxs_nudge[elnode[~fp,:][idxs,:]]=1

            idxs=np.where(idxs_nudge == 1)[0]
            global_idxs[i] = idxs


        #logger.info(f'len of nudge idxs is {len(idxs)}')
        logger.info(f'It took {time() -t0} sencods to calcuate nudge coefficient')

        nudge = [f"rlmax={rlmax}, rnu_day={rnu_day}"]
        nudge.extend("\n")
        nudge.append(f"{NE} {NP}")
        nudge.extend("\n")
        hgrid = self.hgrid.to_dict()
        nodes = hgrid['nodes']
        elements = hgrid['elements']
        for idn, (coords, values) in nodes.items():
            line = [f"{idn}"]
            line.extend([f"{x:<.7e}" for x in coords])
            line.extend([f"{nudge_coeff[int(idn)-1]:<.7e}"])
            line.extend("\n")
            nudge.append(" ".join(line))

        for id, element in elements.items():
            line = [f"{id}"]
            line.append(f"{len(element)}")
            line.extend([f"{e}" for e in element])
            line.extend("\n")
            nudge.append(" ".join(line))

        with open(outdir / 'TEM_nudge.gr3','w+') as fid:
            fid.writelines(nudge)

        shutil.copy2(outdir / 'TEM_nudge.gr3', outdir / 'SAL_nudge.gr3')

        return global_idxs

    def fetch_data(
        self, outdir: Union[str, os.PathLike], vgrid, start_date, rnday, restart=False, rlmax=None, rnu_day=None,
        archive_netcdf=False,archivedir=None
    ):
        """
        fetch data from the database and generate nudge file
        see gen_nudge for the meaning of rlmax, rnu_day
        """

        outdir = pathlib.Path(outdir)
        if archive_netcdf:  # create new download dir
            if archivedir is None:
                archivedir = outdir
            archivedir = pathlib.Path(f'{archivedir}')
            archivedir.mkdir(exist_ok=True,parents=True)     

        self.start_date = start_date
        self.rnday=rnday
        self.timevector=np.arange(
            self.start_date,
            self.start_date + timedelta(days=self.rnday+1),
            timedelta(days=1)).astype(datetime)

        if isinstance(vgrid,os.PathLike):
            vd=Vgrid.open(vgrid)
        elif isinstance(vgrid,Vgrid):
            vd = vgrid
        else:
            raise('Incorrect vgrid input -- Nudge.fetch_data needs vgrid as a type os.PathLike or pyschism.mesh.Vgrid')

        sigma=vd.sigma
        sigma[np.isnan(sigma)]=-1
        #define nudge zone and strength
        rlmax = 1.5 if rlmax is None else rlmax
        rnu_day = 0.25 if rnu_day is None else rnu_day
        logger.info(f'Max relax distance is {rlmax} degree, max relax strengh is {rnu_day} days.')
        #Get the index for nudge
        global_idxs = self.gen_nudge(outdir, rlmax = rlmax, rnu_day=rnu_day)

        #get bathymetry
        depth = self.hgrid.values

        #compute zcor
        zcor = depth[:,None]*sigma
        nvrt=zcor.shape[1]

        #allocate output variables
        include = np.concatenate([global_idxs[i] for i in self.ocean_bnd_ids])

        nNode = include.shape[0]
        one = 1
        ntimes = self.rnday+1

        #timeseries_s=np.zeros([ntimes,nNode,nvrt,one])
        #timeseries_t=np.zeros([ntimes,nNode,nvrt,one])
        #ndt=np.zeros([ntimes])
        if restart:
            dst_temp = Dataset(outdir / 'TEM_nu.nc', 'a', format='NETCDF4')
            dst_salt = Dataset(outdir / 'SAL_nu.nc', 'a', format='NETCDF4')
            time_idx_restart = dst_temp['time'][:].shape[0]
        else:
            dst_temp = Dataset(outdir / 'TEM_nu.nc', 'w', format='NETCDF4')
            #dimensions
            dst_temp.createDimension('node', nNode)
            dst_temp.createDimension('nLevels', nvrt)
            dst_temp.createDimension('one', one)
            dst_temp.createDimension('time', None)
            #variables
            dst_temp.createVariable('time', 'f', ('time',))
            #dst_temp['time'][:] = ndt

            dst_temp.createVariable('map_to_global_node', 'i4', ('node',))
            dst_temp['map_to_global_node'][:] = include+1

            dst_temp.createVariable('tracer_concentration', 'f', ('time', 'node', 'nLevels', 'one'))
            #dst_temp['tracer_concentration'][:,:,:,:] = timeseries_t

            #salinity
            dst_salt = Dataset(outdir / 'SAL_nu.nc', 'w', format='NETCDF4')
            #dimensions
            dst_salt.createDimension('node', nNode)
            dst_salt.createDimension('nLevels', nvrt)
            dst_salt.createDimension('one', one)
            dst_salt.createDimension('time', None)
            #variables
            dst_salt.createVariable('time', 'f', ('time',))
            #dst_salt['time'][:] = ndt

            dst_salt.createVariable('map_to_global_node', 'i4', ('node',))
            dst_salt['map_to_global_node'][:] = include+1

            dst_salt.createVariable('tracer_concentration', 'f', ('time', 'node', 'nLevels', 'one'))
            #dst_salt['tracer_concentration'][:,:,:,:] = timeseries_s

        logger.info('**** Accessing GOFS data*****')
        if restart:
            #restart from one day earlier to make sure all files consistant
            timevector = self.timevector[time_idx_restart-1:]
            it0 = time_idx_restart-1
        else:
            timevector = self.timevector
            it0 = 0

        t0=time()
        for it1, date in enumerate(timevector):

            it = it0 + it1

            database=get_database(date)
            logger.info(f'Fetching data for {date} from database {database}')
            print(f'Fetching data for {date} from database {database}')

            ind1 = 0
            ind2 = 0
            for ibnd in self.ocean_bnd_ids:
                include = global_idxs[ibnd]

                ind1 = ind2
                ind2 = ind1 + include.shape[0]
                #Get open nudge array
                nlon = self.hgrid.coords[include, 0]
                nlat = self.hgrid.coords[include, 1]
                nlonc = nlon.mean()
                nlatc = nlat.mean()
                xi,yi = transform_ll_to_cpp(nlon, nlat, nlonc, nlatc)
                bxy = np.c_[yi, xi]

                zcor2=zcor[include,:]
                idxs=np.where(zcor2 > 5000)
                zcor2[idxs]=5000.0-1.0e-6

                #construct schism grid
                x2i=np.tile(xi,[nvrt,1]).T
                y2i=np.tile(yi,[nvrt,1]).T
                bxyz=np.c_[zcor2.reshape(np.size(zcor2)),y2i.reshape(np.size(y2i)),x2i.reshape(np.size(x2i))]
                logger.info('Computing SCHISM zcor is done!')

                xmin = floor_2_decimals(np.min(nlon))
                xmax = ceil_2_decimals(np.max(nlon))
                ymin= floor_2_decimals(np.min(nlat))
                ymax = ceil_2_decimals(np.max(nlat))
                bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)
                logger.info(f'bbox for nudge is {bbox}')

                time_idx, lon_idx1, lon_idx2, lat_idx1, lat_idx2, x2, y2, _ = get_idxs(date, database, bbox, lonc=nlonc, latc=nlatc)

                if date >= datetime.utcnow():
                    date2 = datetime.utcnow() - timedelta(days=1)
                    url = f'https://tds.hycom.org/thredds/dodsC/{database}/FMRC/runs/GLBy0.08_930_FMRC_RUN_' + \
                        f'{date2.strftime("%Y-%m-%dT12:00:00Z")}?depth[0:1:-1],lat[{lat_idx1}:1:{lat_idx2}],' + \
                        f'lon[{lon_idx1}:1:{lon_idx2}],time[{time_idx}],' + \
                        f'water_temp[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'salinity[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}]'
                else:
                    url=f'https://tds.hycom.org/thredds/dodsC/{database}?lat[{lat_idx1}:1:{lat_idx2}],' + \
                        f'lon[{lon_idx1}:1:{lon_idx2}],depth[0:1:-1],time[{time_idx}],' + \
                        f'water_temp[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}],' + \
                        f'salinity[{time_idx}][0:1:39][{lat_idx1}:1:{lat_idx2}][{lon_idx1}:1:{lon_idx2}]'
            
                # Generate a local copy of the HYCOM file for future use
                if archive_netcdf:
                    hycom_fname = archivedir / f'hycom_{date.strftime("%Y%m%d_%H")}_{xmin:0.3f}_{xmax:0.3f}E__{ymin:0.3f}_{ymax:0.3f}N.nc'
                    if not hycom_fname.exists():
                        logger.info(f'Local copy of HYCOM file {hycom_fname} does not exist.')
                        logger.info(f'Downloading HYCOM data from {database} ... ')
                        xr.open_dataset(url).to_netcdf(hycom_fname)
                        logger.info(f'Data saved to {hycom_fname.resolve()}')
                    try:
                        logger.info(f'Reading HYCOM file: {hycom_fname}')
                        ds=Dataset(hycom_fname)
                    except:
                        logger.info(f'Failed to open local copy of HYCOM file {hycom_fname} ... Downloading...')
                        ds=Dataset(url)
                else:
                    logger.info(f'Downloading HYCOM data from {database} ... ')
                    ds=Dataset(url)

                salt=np.squeeze(ds['salinity'][:,:,:])
                temp=np.squeeze(ds['water_temp'][:,:,:])

                #Convert temp to potential temp
                dep=ds['depth'][:]
                ptemp = ConvertTemp(salt, temp, dep)

                logger.info(f'****Interpolation starts for boundary {ibnd}****')

                #ndt[it]=it
                #salt
                dst_salt['time'][it] = it
                logger.info(f'interp_to_points_3d(dep, y2, x2, bxyz, salt)')
                salt_int = interp_to_points_3d(dep, y2, x2, bxyz, salt)
                salt_int = salt_int.reshape(zcor2.shape)
                #timeseries_s[it,:,:,0]=salt_int
                dst_salt['tracer_concentration'][it,ind1:ind2,:,0] = salt_int

                #temp
                dst_temp['time'][it] = it
                logger.info(f'interp_to_points_3d(dep, y2, x2, bxyz, ptemp)')
                temp_int = interp_to_points_3d(dep, y2, x2, bxyz, ptemp)
                temp_int = temp_int.reshape(zcor2.shape)
                #timeseries_t[it,:,:,0]=temp_int
                dst_temp['tracer_concentration'][it,ind1:ind2,:,0] = temp_int

                ds.close()

        #dst_temp.close()
        #dst_salt.close()

        logger.info(f'Writing *_nu.nc takes {time()-t0} seconds')

class DownloadHycom:

    """
        Generate ocean boundary condition data by:

        1. download data with fetch_data using fmt='schism'
    
        2. Compile and run "schism/src/Utility/Gen_Hotstart/gen_3Dth_from_hycom.f90" in schism repo,
                expects (1) hgrid.gr3; (2) hgrid.ll; (3) vgrid.in;'

        Or:
        
        1. just download hycom data with fmt='hycom'    

    """

    def __init__(self, hgrid=None, bbox=None):

        if hgrid is None and bbox is None:
            print('Either hgrid or bbox must be provided to use fetch_data()')

        if hgrid is not None:
            xmin, xmax = hgrid.coords[:, 0].min(), hgrid.coords[:, 0].max()
            ymin, ymax = hgrid.coords[:, 1].min(), hgrid.coords[:, 1].max()
            self.bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)
        elif bbox is not None:
            self.bbox = bbox

    def fetch_data(self, start_date, rnday=1, fmt='schism', bnd=False, nudge=False, time_stride=8, lon_stride=1, lat_stride=1,outdir='./', overwrite=False):
        '''
        start_date: datetime.datetime
        rnday: integer
        fmt: 'schism' - for Fortran code; 'hycom' - raw netCDF from HYCOM
        bnd: file names are SSH_*.nc, TS_*.nc, UV_*.nc used in gen_hot_3Dth_from_hycom.f90
        nudge: file name is TS_*.nc used in gen_nudge_from_hycom.f90
        outdir: directory for output files
        lon_stide,lat_stide : integer, stride along lon and lat dimension to sub sample data (or not)
        time_stride: integer
            Stride along HYCOM time dimension within each daily file.
            time_stride=1 downloads all times; 2 would take every other timestep, etc.

        Example: fmt='schism 

        netcdf hycom_20041230 {
        dimensions:
                depth = 40 ;
                lat = 223 ;
                lon = 390 ;
                time = 1 ;
        variables:
                double depth(depth) ;
                        depth:_FillValue = NaN ;
                        depth:long_name = "Depth" ;
                        depth:standard_name = "depth" ;
                        depth:units = "m" ;
                        depth:positive = "down" ;
                        depth:axis = "Z" ;
                        depth:NAVO_code = 5 ;
                double lat(lat) ;
                        lat:_FillValue = NaN ;
                        lat:long_name = "Latitude" ;
                        lat:standard_name = "latitude" ;
                        lat:units = "degrees_north" ;
                        lat:axis = "Y" ;
                        lat:NAVO_code = 1 ;
                double lon(lon) ;
                        lon:_FillValue = NaN ;
                        lon:long_name = "Longitude" ;
                        lon:standard_name = "longitude" ;
                        lon:units = "degrees_east" ;
                        lon:point_spacing = "even" ;
                        lon:modulo = "360 degrees" ;
                        lon:axis = "X" ;
                        lon:NAVO_code = 2 ;
                double time(time) ;
                        time:_FillValue = NaN ;
                        time:long_name = "Valid Time" ;
                        time:time_origin = "2000-01-01 00:00:00" ;
                        time:axis = "T" ;
                        time:NAVO_code = 13 ;
                        time:units = "hours since 2000-01-01" ;
                        time:calendar = "gregorian" ;
                short water_u(time, depth, lat, lon) ;
                        water_u:_FillValue = -30000s ;
                        water_u:long_name = "Eastward Water Velocity" ;
                        water_u:standard_name = "eastward_sea_water_velocity" ;
                        water_u:units = "m/s" ;
                        water_u:NAVO_code = 17 ;
                        water_u:add_offset = 0.f ;
                        water_u:scale_factor = 0.001f ;
                        water_u:missing_value = -30000s ;
                short water_v(time, depth, lat, lon) ;
                        water_v:_FillValue = -30000s ;
                        water_v:long_name = "Northward Water Velocity" ;
                        water_v:standard_name = "northward_sea_water_velocity" ;
                        water_v:units = "m/s" ;
                        water_v:NAVO_code = 18 ;
                        water_v:add_offset = 0.f ;
                        water_v:scale_factor = 0.001f ;
                        water_v:missing_value = -30000s ;
                short water_temp(time, depth, lat, lon) ;
                        water_temp:_FillValue = -30000s ;
                        water_temp:long_name = "Water Temperature" ;
                        water_temp:standard_name = "sea_water_temperature" ;
                        water_temp:units = "degC" ;
                        water_temp:NAVO_code = 15 ;
                        water_temp:comment = "in-situ temperature" ;
                        water_temp:add_offset = 20.f ;
                        water_temp:scale_factor = 0.001f ;
                        water_temp:missing_value = -30000s ;
                short salinity(time, depth, lat, lon) ;
                        salinity:_FillValue = -30000s ;
                        salinity:long_name = "Salinity" ;
                        salinity:standard_name = "sea_water_salinity" ;
                        salinity:units = "psu" ;
                        salinity:NAVO_code = 16 ;
                        salinity:add_offset = 20.f ;
                        salinity:scale_factor = 0.001f ;
                        salinity:missing_value = -30000s ;
                short surf_el(time, lat, lon) ;
                        surf_el:_FillValue = -30000s ;
                        surf_el:long_name = "Water Surface Elevation" ;
                        surf_el:standard_name = "sea_surface_elevation" ;
                        surf_el:units = "m" ;
                        surf_el:NAVO_code = 32 ;
                        surf_el:add_offset = 0.f ;
                        surf_el:scale_factor = 0.001f ;
                        surf_el:missing_value = -30000s ;

        // global attributes:
                        :classification_level = "UNCLASSIFIED" ;
                        :distribution_statement = "Approved for public release. Distribution unlimited." ;
                        :downgrade_date = "not applicable" ;
                        :classification_authority = "not applicable" ;
                        :institution = "Naval Oceanographic Office" ;
                        :source = "HYCOM archive file" ;
                        :history = "archv2ncdf3z" ;
                        :field_type = "instantaneous" ;
        

        '''

        # --- build list of days (one output file per day)
        if rnday == 1:
            timevector = [start_date]
        else:
            timevector = np.arange(start_date, start_date+timedelta(days=rnday), timedelta(days=1)).astype(datetime)

        coord_cache={}
        for i, date in enumerate(tqdm(timevector, desc="Downloading HYCOM daily files", unit="day")):
            
            tqdm.write(f"Starting {date:%Y-%m-%d}")

            # Normalize to midnight for "daily" files
            day_start = datetime(date.year, date.month, date.day)
            day_end   = day_start + timedelta(days=1)

            foutname = pathlib.Path(outdir) / f'hycom_{day_start.strftime("%Y%m%d")}.nc'

            if foutname.exists() and not overwrite:
                print(f'{foutname} exists and overwrite=False ... skipping ...')
                continue

            database = get_database(day_start)
            logger.info(f'Fetching data for {day_start} from database {database}')
            print(f'Fetching data for {day_start} from database {database}')

            # time_idx, lon_idx1, lon_idx2, lat_idx1, lat_idx2, x2, y2, isLonSame = get_idxs(day_start, database, self.bbox)
            #
            # # --- get full time range for day
            # t_start, t_end = get_time_range_indices(database, day_start, day_end)
            # if t_start is None:
            #     logger.warning(f'No HYCOM times found for {day_start} in {database} ... skipping ...')
            #     print(f'No HYCOM times found for {day_start} in {database} ... skipping ...')
            #     continue
            #
            # # OPeNDAP time slice string (inclusive end index)
            # tsel = f'{t_start}:{time_stride}:{t_end}'

            database_idx = get_idxs_and_time_range(day_start, database, self.bbox, cache=coord_cache)
            t_start=database_idx[0]
            t_end=database_idx[1]
            lon_idx1=database_idx[2]
            lon_idx2=database_idx[3]
            lat_idx1=database_idx[4]
            lat_idx2=database_idx[5]
            x2=database_idx[6]
            y2=database_idx[7]
            isLonSame=database_idx[8]
            day_start_used=database_idx[9]

            if t_start is None:
                print(f'No HYCOM data for {day_start} ... skipping ...')
                continue

            # if missing and shifted forward, you may want the filename to follow the actual data day:
            foutname = pathlib.Path(outdir) / f'hycom_{day_start_used.strftime("%Y%m%d")}.nc'

            tsel = f'{t_start}:{time_stride}:{t_end}'

            url_ssh = (
                f'https://tds.hycom.org/thredds/dodsC/{database}?'
                f'lat[{lat_idx1}:{lat_stride}:{lat_idx2}],'
                f'lon[{lon_idx1}:{lon_stride}:{lon_idx2}],'
                f'depth[0:1:-1],'
                f'time[{tsel}],'
                f'surf_el[{tsel}][{lat_idx1}:{lat_stride}:{lat_idx2}][{lon_idx1}:{lon_stride}:{lon_idx2}],'
                f'water_u[{tsel}][0:1:39][{lat_idx1}:{lat_stride}:{lat_idx2}][{lon_idx1}:{lon_stride}:{lon_idx2}],'
                f'water_v[{tsel}][0:1:39][{lat_idx1}:{lat_stride}:{lat_idx2}][{lon_idx1}:{lon_stride}:{lon_idx2}],'
                f'water_temp[{tsel}][0:1:39][{lat_idx1}:{lat_stride}:{lat_idx2}][{lon_idx1}:{lon_stride}:{lon_idx2}],'
                f'salinity[{tsel}][0:1:39][{lat_idx1}:{lat_stride}:{lat_idx2}][{lon_idx1}:{lon_stride}:{lon_idx2}]'
            )

            if fmt == 'schism':

                logger.info(f'filename is {foutname}')

                ds = xr.open_dataset(url_ssh)

                # convert in-situ temperature to potential temperature
                temp = ds.water_temp.values
                salt = ds.salinity.values
                dep  = ds.depth.values

                ptemp = ConvertTemp(salt, temp, dep)

                # drop water_temp variable and add new temperature variable
                ds = ds.drop('water_temp')
                ds['temperature'] = (['time', 'depth', 'lat', 'lon'], ptemp)
                ds.temperature.attrs = {
                    'long_name': 'Sea water potential temperature',
                    'standard_name': 'sea_water_potential_temperature',
                    'units': 'degC'
                }

                if not isLonSame:
                    logger.info('Lon is not the same!')
                    ds = convert_longitude(ds, self.bbox)

                ds = ds.rename_dims({'lon': 'xlon'})
                ds = ds.rename_dims({'lat': 'ylat'})
                ds = ds.rename_vars({'lat': 'ylat'})
                ds = ds.rename_vars({'lon': 'xlon'})

                t0 = time()
                logger.info('Start writing nc file!')
                ds.to_netcdf(
                    foutname, 'w',
                    unlimited_dims='time',
                    encoding={ 
                        'temperature': {
                            'dtype': 'h',
                            '_FillValue': -30000.,
                            'scale_factor': 0.001,
                            'add_offset': 20.,
                            'missing_value': -30000.
                        }
                    }
                )
                ds.close()
                logger.info(f'It took {time() - t0} seconds to write nc file')

            elif fmt == 'hycom':

                ds = xr.open_dataset(url_ssh)
                ds.to_netcdf(foutname, 'w')
                ds.close()

        if fmt == 'schism':
            self.make_sym_links(outdir=outdir, timevector=timevector, bnd=bnd, nudge=nudge)

    def make_sym_links(self, outdir=pathlib.Path('./'), datadir=None, start_date=None, rnday=None, timevector=None, bnd=True, nudge=False):
        
        if timevector is None:
            if rnday == 1:
                timevector = [start_date]
            else:
                timevector = np.arange(
                    start_date, start_date + timedelta(days=rnday+1), timedelta(days=1)
                ).astype(datetime)

        for i, date in enumerate(timevector):

            if datadir is None:
                src_dir = pathlib.Path(outdir).resolve()
                dst_dir = pathlib.Path(outdir).resolve()
            else:
                src_dir = pathlib.Path(datadir).resolve()
                dst_dir = pathlib.Path(outdir).resolve()
 
            foutname = pathlib.Path(src_dir) / f'hycom_{date.strftime("%Y%m%d")}.nc'

            logger.info(f'src dir is {src_dir}')
            logger.info(f'dst dir is {dst_dir}')

            src = dst_dir / foutname

            if not src.exists():
                raise FileNotFoundError(f"Source file does not exist: {src}")

            if bnd:
                names = ['SSH', 'TS', 'UV']
                for name in names:
                    dst = dst_dir / f"{name}_{i+1}.nc"
                    # remove existing symlink or file at destination
                    if dst.is_symlink():
                        dst.unlink()
                    elif dst.exists():
                        raise FileExistsError(f"{dst} exists and is not a symlink; refusing to overwrite.")
                    os.symlink(src, dst)   # src/dst are Path-like; works on modern Python
            elif nudge:
                dst = dst_dir / f"TS_{i+1}.nc"
                if dst.is_symlink():
                    dst.unlink()
                elif dst.exists():
                    raise FileExistsError(f"{dst} exists and is not a symlink; refusing to overwrite.")
                os.symlink(src, dst)
