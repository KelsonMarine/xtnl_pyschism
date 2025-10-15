from enum import Enum
import os
import pathlib
from typing import Union
import numpy as np

from pyproj import CRS  # type: ignore[import]
from shapely.geometry import Polygon, MultiPolygon, Point
import geopandas as gpd

from pyschism.mesh.base import Gr3
from pyschism.mesh.parsers import grd


class FrictionFilename(Enum):
    MANNINGS_N = 'manning.gr3'
    DRAG_COEFFICIENT = 'drag.gr3'
    ROUGHNESS_LENGTH = 'rough.gr3'

    @classmethod
    def _missing_(self, name):
        raise ValueError(f'{name} is not a valid filename for a friction '
                         'file.')


class NchiType(Enum):
    MANNINGS_N = -1
    ROUGHNESS_LENGTH = 1
    DRAG_COEFFICIENT = 0


class Fgrid(Gr3):
    """
    Base class for all friction types (e.g. manning.grd, drag.grd, etc...)
    """

    def __init__(self, nchi: NchiType, *argv, **kwargs):
        self._nchi = nchi
        self._fname = FrictionFilename[NchiType(nchi).name]
        super().__init__(*argv, **kwargs)

    @property
    def nchi(self):
        return self._nchi.value

    @property
    def fname(self):
        return self._fname.value

    @classmethod
    def open(cls, file: Union[str, os.PathLike],
             crs: Union[str, CRS] = None):
        filename = pathlib.Path(file).name
        if cls.__name__ == "Fgrid":
            return FrictionDispatch[
                FrictionFilename(filename).name].value(
                    **grd.read(pathlib.Path(file), boundaries=False, crs=crs))
        else:
            return super().open(file, crs)

    @classmethod
    def constant(cls, hgrid, value):
        obj = cls(**{k: v for k, v in hgrid.to_dict().items() if k
                     in ['nodes', 'elements', 'description', 'crs']})
        obj.values[:] = value
        obj.description = f'{cls.__name__.lower()} {obj.crs}'
        return obj

    def add_region(
            self,
            region: Union[Polygon, MultiPolygon],
            value
     ):
        # Assuming input polygons are in EPSG:4326
        if isinstance(region, Polygon):
            region = [region]
        gdf1 = gpd.GeoDataFrame(
                {'geometry': region}, crs=self.crs)

        points = [Point(*coord) for coord in self.coords]
        gdf2 = gpd.GeoDataFrame(
                 {'geometry': points, 'index': list(range(len(points)))},
                crs=self.crs)
        gdf_in = gpd.sjoin(gdf2, gdf1, predicate="within")
        picks = ([i.index for i in gdf_in.itertuples()])
        self.values[picks] = value

    def modify_by_region(self, hgrid, fname, value, depth1, flag):
        '''
        reset (flag==0) or add (flag==1) value to a region
        '''
        lines=[line.strip().split() for line in open(fname, 'r').readlines()]
        data=np.squeeze(np.array([lines[3:]])).astype('float')
        x=data[:,0]
        y=data[:,1]
        coords = list( zip(x, y))
        poly = Polygon(coords)

        # Assuming input polygons are in EPSG:4326
        #if isinstance(region, Polygon):
        #    region = [region]
        gdf1 = gpd.GeoDataFrame(
                {'geometry': [poly]}, crs=self.crs)

        points = [Point(*coord) for coord in self.coords]
        gdf2 = gpd.GeoDataFrame(
                 {'geometry': points, 'index': list(range(len(points)))},
                crs=self.crs)
        gdf_in = gpd.sjoin(gdf2, gdf1, predicate="within")
        picks = [i.index for i in gdf_in.itertuples()]
        if flag == 0:
            self.values[picks] = value
        else:
            picks2 = np.where(-hgrid.values > depth1)
            picks3 = np.intersect1d(picks, picks2)
            self.values[picks3] = self.values[picks3] + value

    @classmethod
    def phi_to_fgrid(cls,
                    hgrid: Gr3,
                    lon: np.ndarray, 
                    lat: np.ndarray, 
                    phi: np.ndarray,
                    out="z0",              # "z0" or "Cd" (inferred if called from subclass)
                    method="nearest",      # "nearest" or "idw"
                    k=8,                   # neighbors for IDW
                    power=2.0,             # IDW power
                    ks_factor=2.5,         # k_s ≈ ks_factor * d50
                    z0_clip=(1e-6, 1e-2),  # meters
                    kappa=0.41,            # von Kármán
                    hmin=1.0,              # min depth for Cd (m)
                    cd_clip=(1e-4, 5e-2),  # plausible Cd range
                    ):
        """
        Map scattered (lon, lat, phi) to mesh nodes and compute z0 or Cd.

        Parameters
        ----------
        lon, lat, phi : 1D arrays
            Scattered points in degrees (lon, lat) and Krumbein phi.
            phi relates to grain size via D_mm = 2^(-phi).
        out : {"z0","Cd"}
            Which field to return. If return_both=True, returns (z0, Cd).
        method : {"nearest","idw"}
            Interpolation method from points to nodes (projected XY).
        k : int
            Number of neighbors for IDW (ignored for nearest).
        power : float
            IDW power (2.0 typical).
        ks_factor : float
            Nikuradse sand-grain roughness multiplier on d50 (k_s = ks_factor*d50).
        z0_clip : (float, float)
            Min/max clip for z0 in meters.
        kappa : float
            von Kármán constant.
        hmin : float
            Minimum depth used when converting z0toCd to avoid singularities.
        cd_clip : (float, float)
            Min/max clip for Cd.
        return_both : bool
            If True, return (z0_nodes, Cd_nodes) regardless of `out`.

        Returns
        -------
        out_arr : np.ndarray
            Node-wise array of z0 (m) or Cd (dimensionless).
            If return_both=True, returns (z0_nodes, Cd_nodes).
        """
        from scipy.spatial import cKDTree
        from pyproj import CRS, Transformer

        if 'RoughnessLength' in cls.__name__ or issubclass(cls, RoughnessLength):
            out = "z0"
        elif 'DragCoefficient' in cls.__name__ or issubclass(cls, DragCoefficient):
            out = "Cd"

        obj = cls.constant(hgrid, np.nan)

        # ---- 0) Pull nodes from hgrid ----
        coords = hgrid.coords
        assert (hgrid.crs == 'EPGS:4326') or (hgrid.crs == 'WGS84')
        depth = hgrid.values

        # ---- 1) Convert phi to d50 (m) to k_s to z0 (m) ----
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        phi = np.asarray(phi)

        ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(phi)
        if not np.any(ok):
            raise ValueError("No valid (lon, lat, phi) samples provided.")
        lonp, latp, phip = lon[ok], lat[ok], phi[ok]

        d_mm = 2.0 ** (-phip)           # Krumbein phi to mm
        d50  = d_mm / 1000.0            # mm to meters
        ks   = ks_factor * d50          # Nikuradse roughness
        z0_pts = ks / 30.0              # hydraulic roughness length
        z0_pts = np.clip(z0_pts, *z0_clip)

        # ---- 2) Project (lon,lat) to local metric coordinates (m) ----
        coords0 = float(np.nanmean(coords[:,0]))
        coords1 = float(np.nanmean(coords[:,1]))
        wgs84 = CRS.from_epsg(4326)

         # Azimuthal Equidistant projection centred on origin point
        aeqd  = CRS.from_proj4(f"+proj=aeqd +lat_0={coords1} +lon_0={coords0} +datum=WGS84 +units=m +no_defs")
        to_xy = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform

        xp, yp = to_xy(lonp,      latp)
        xn, yn = to_xy(coords[:,0], coords[:,1])

        # ---- 3) Interpolate z0 from points to nodes ----
        tree = cKDTree(np.c_[xp, yp])
        if method == "nearest":
            _, idx = tree.query(np.c_[xn, yn], k=1)
            z0_nodes = z0_pts[idx]
        elif method == "idw":
            k_eff = min(k, len(z0_pts))
            dist, idx = tree.query(np.c_[xn, yn], k=k_eff)
            # Ensure 2D arrays
            dist = np.atleast_2d(dist)
            idx  = np.atleast_2d(idx)
            w = 1.0 / np.maximum(dist, 1e-12) ** power
            z0_nodes = (w * z0_pts[idx]).sum(axis=1) / w.sum(axis=1)
        else:
            raise ValueError("method must be 'nearest' or 'idw'.")

        z0_nodes = np.clip(z0_nodes, *z0_clip)

        # ---- 4) z0 to Cd using local depth ----
        h = np.maximum(np.asarray(depth), float(hmin))
        # Avoid log singularities
        ratio = np.maximum(h / np.maximum(z0_nodes, z0_clip[0]), 1.0000001)
        Cd_nodes = (kappa / np.log(ratio)) ** 2
        Cd_nodes = np.clip(Cd_nodes, *cd_clip)

        if out == "z0":
            obj.values[:] = z0_nodes
        elif out == "Cd":
            obj.values[:] = Cd_nodes

        return obj
        

class ManningsN(Fgrid):
    """  Class for representing Manning's n values.  """

    def __init__(self, *argv, **kwargs):
        self.hmin_man = 1.
        super().__init__(NchiType.MANNINGS_N, *argv, **kwargs)

    @classmethod
    def linear_with_depth(
            cls,
            hgrid: Union[str, os.PathLike, Gr3],
            min_value: float = 0.02,
            max_value: float = 0.05,
            min_depth: float = None,
            max_depth: float = None):

        # Inspired by https://github.com/schism-dev/schism/blob/master/src/Utility/Pre-Processing/NWM/Manning/write_manning.py
        obj = cls.constant(hgrid, np.nan)
        min_depth = np.min(-hgrid.values) if min_depth is None \
            else float(min_depth)
        max_depth = np.max(-hgrid.values) if max_depth is None \
            else float(max_depth)

        values = (
                min_value + (-hgrid.values - min_depth)
                * (max_value - min_value) / (max_depth - min_depth))

        if min_value is not None:
            values[values < min_value] = min_value

        if max_value is not None:
            values[values > max_value] = max_value

        obj.values[:] = values

        return obj


class RoughnessLength(Fgrid):

    def __init__(self, *argv, **kwargs):
        self.dzb_min = 0.5
        self.dzb_decay = 0.
        super().__init__(NchiType.ROUGHNESS_LENGTH, *argv, **kwargs)


class DragCoefficient(Fgrid):

    def __init__(self, *argv, **kwargs):
        super().__init__(NchiType.DRAG_COEFFICIENT, *argv, **kwargs)

    @classmethod
    def linear_with_depth(
            cls,
            hgrid: Union[str, os.PathLike, Gr3],
            depth1: float = -1.0,  # Are depth1 and depth2 positive up or positive down?
            depth2: float = -3.0,
            bfric_river: float = 0.0025,
            bfric_land: float = 0.025
    ):

        obj = cls.constant(hgrid, np.nan)

        values = (bfric_river + (depth1 + hgrid.values) *
                  (bfric_land - bfric_river) / (depth1-depth2))
        values[values > bfric_land] = bfric_land
        values[values < bfric_river] = bfric_river

        obj.values[:] = values

        return obj


class FrictionDispatch(Enum):
    MANNINGS_N = ManningsN
    DRAG_COEFFICIENT = DragCoefficient
    ROUGHNESS_LENGTH = RoughnessLength
