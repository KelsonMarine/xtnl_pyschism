from abc import ABC
from datetime import datetime, timedelta
from functools import lru_cache
import logging
import os
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, MultiPoint
from collections import defaultdict
from pyproj import CRS, Transformer
import pytz
from scipy.interpolate import interp1d
from shapely import ops

from pyschism import dates
from pyschism.mesh.base import Gr3
from pyschism.mesh.hgrid import Hgrid


logger = logging.getLogger(__name__)


class SourceSinkDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    @property
    def elements(self):
        if not hasattr(self, "_elements"):
            unique_elements = set()
            for elements in self.data.values():
                for element_id in elements.keys():
                    unique_elements.add(element_id)
            self._elements = list(
                map(str, sorted(list(map(int, unique_elements)))))
            # self._elements = list(
            #     map(int, sorted(list(map(int, unique_elements)))))
        return self._elements

    @property
    def timevector(self):
        if not hasattr(self, '_timevector'):
            self._timevector = list(sorted(self.data.keys()))
        return self._timevector

    @property
    def df(self):
        if not hasattr(self, "_df"):
            data = []
            for time, edata in self.data.items():
                for eid, _edata in edata.items():
                    data.append({"time": time, "element_id": eid, **_edata})
            self._df = pd.DataFrame(data)
        return self._df

    @property
    def data(self):
        return self._data


class Sources(SourceSinkDataset):
    def __init__(self, data):
        for time, edata in data.items():
            assert isinstance(time, datetime)
            for eid, datapoint in edata.items():
                assert (
                    datapoint["flow"] >= 0.0
                ), f"Invalid source point for element_id={eid} during time {str(time)}. "
                f'Sources must be >= 0 but got value of {datapoint["flow"]}.'
                assert isinstance(datapoint["temperature"], float)
                assert isinstance(datapoint["salinity"], float)
        super().__init__(data)


class Sinks(SourceSinkDataset):
    def __init__(self, data):
        for time, edata in data.items():
            assert isinstance(time, datetime)
            for eid, datapoint in edata.items():
                assert (
                    datapoint["flow"] <= 0.0
                ), f"Invalid sink point for element_id={eid} during time {str(time)}. "
                f'Sinks must be <= 0 but got value of {datapoint["flow"]}.'
        super().__init__(data)


class TimeHistoryFile(ABC):
    def __init__(self, source_sink: SourceSinkDataset, start_date, rnday, filename):
        self.dataset = source_sink
        self.filename = filename
        self.start_date = start_date
        self.rnday = rnday

    def __str__(self):
        logger.info(
            f'Generate {self.__class__.__name__.lower()} time history string.')
        start = datetime.now()
        # build ts matrix
        ts_matrix = np.full((len(self.dataset.timevector),
                            len(self.dataset.elements)), np.nan)

        for i, element_id in enumerate(self.dataset.elements):
            ts_matrix[:, i] = self.get_element_timeseries(element_id)

        if np.any(np.any(np.isnan(ts_matrix), axis=0)):
            # handle irregular datasets (untested)
            for column_index in np.where(np.any(np.isnan(ts_matrix), axis=0)):
                finite = np.where(np.isfinite(ts_matrix[:, column_index]))[0]
                ts_matrix[:finite[0], column_index] = 0.
                ts_matrix[finite[-1]:, column_index] = 0.
                fit = interp1d(np.array(self.dataset.timevector)[
                               finite], ts_matrix[finite, column_index])
                non_finite = np.where(np.isfinite(
                    ts_matrix[:, column_index]))[0]
                ts_matrix[non_finite, column_index] = fit(
                    np.array(self.dataset.timevector)[non_finite])

        data = []
        for i, row in enumerate(ts_matrix):
            try:
                relative_time = (self.dataset.timevector[i] - self.start_date).total_seconds()
            except:
                if i==1:
                    print('[WARNING] Assumming dataset.timevector timezone = pytz.utc')
                relative_time = (dates.localize_datetime(self.dataset.timevector[i]).astimezone(pytz.utc) - self.start_date).total_seconds()

            if relative_time < 0:
                continue
            data.append(" ".join([
                f"{relative_time:G}",
                *[f'{x:.4e}' for x in row]
            ]))
        logger.info(
            f'Generate time history string took {datetime.now() - start}.')
        return "\n".join(data)

    def get_element_timeseries(self, element_id):
        values = []
        for time in self.dataset.timevector:
            values.append(self.dataset.data[time].get(
                element_id, {}).get('flow', np.nan))
        return np.array(values)

    def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
        path = pathlib.Path(path)
        if (path / self.filename).exists() and overwrite is not True:
            raise IOError("File exists and overwrite is not True.")
        with open(path / self.filename, "w") as f:
            f.write(str(self))


class Vsource(TimeHistoryFile):
    def __init__(self, sources: Sources, start_date, rnday, filename="vsource.th"):
        super().__init__(sources, start_date, rnday, filename)


class Msource(TimeHistoryFile):
    def __init__(self, sources, start_date, rnday, filename="msource.th"):
        super().__init__(sources, start_date, rnday, filename)

    def __str__(self):
        data = []
        for i, time in enumerate(self.dataset.timevector):
            relative_time = (time - self.start_date).total_seconds()
            if relative_time < 0:
                continue
            line = [f"{relative_time:G}"]
            # index dictionary at each time for each element index
            for element_id in self.dataset.elements:    
                temperature = (self.dataset.data[time].get(element_id, {}).get("temperature", -9999.0))
                line.append(f"{temperature: .4e}")
            for element_id in self.dataset.elements:
                salinity = (self.dataset.data[time].get(element_id, {}).get("salinity", -9999.0))
                line.append(f"{salinity: .4e}")
            data.append(" ".join(line))
        return "\n".join(data)

    def get_element_timeseries(self, element_id):
        temp = []
        salt = []
        for time in self.dataset.timevector:
            temp.append(self.dataset.data[time].get(
                element_id, {}).get('temperature', -9999.0))
            salt.append(self.dataset.data[time].get(
                element_id, {}).get('salinity', -9999.0))
        return np.array(temp), np.array(salt)
    
    # def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
    #     path = pathlib.Path(path)
    #     if (path / self.filename).exists() and overwrite is not True:
    #         raise IOError("File exists and overwrite is not True.")
    #     with open(path / self.filename, "w") as f:
    #         f.write(str(self))


class Vsink(TimeHistoryFile):
    def __init__(self, sinks: Sinks, start_date, rnday, filename="vsink.th"):
        super().__init__(sinks, start_date, rnday, filename)


class SourceSinkWriter:
    def __init__(self, sources: Sources, sinks: Sinks, filename="source_sink.in"):
        self.sources = sources
        self.sinks = sinks
        self.filename = filename

    def __str__(self):
        logger.info('Generate source_sink string')
        start = datetime.now()
        source_id = self.sources.elements
        data = []
        data.append(f"{len(source_id)}")
        for element_id in source_id:
            data.append(f"{element_id}")
        data.append("")
        sink_id = self.sinks.elements
        data.append(f"{len(sink_id)}")
        for element_id in sink_id:
            data.append(f"{element_id}")
        logger.info(
            f'Generate source_sink string took {datetime.now() - start}')
        return "\n".join(data)

    def write(self, path: Union[str, os.PathLike], overwrite: bool = False):
        path = pathlib.Path(path)
        if (path / self.filename).exists() and overwrite is not True:
            raise IOError("File exists and overwrite is not True.")
        with open(path / self.filename, "w") as f:
            f.write(str(self))


class SourceSink:

    def __init__(self, sources: Sources = None, sinks: Sinks = None, data: dict = None):
        """
        Initialize SourceSink object with Sources and Sinks objects. 
        
        __init__ method created by Kelson Marine
        """
        if sources is not None:
            self._sources = sources
        if sinks is not None:
            self._sinks = sinks

        # initialize backing fields
        self._data = data or {}
        self._df = None
        self._start_date = None
        self._rnday = None


    def __add__(self, other):
        source_sink = SourceSink()
        source_sink.sources = Sources(
            {**self.sources.data, **other.sources.data})
        source_sink.sinks = Sinks({**self.sinks.data, **other.sinks.data})
        return source_sink

    def __len__(self):
        return len(self._data)

    def add_data(
        self,
        time: datetime,
        element_id: str,
        flow: float,
        temperature: float = np.nan,
        salinity: float = np.nan,
    ):
        
        '''
        Note: this function fails whent the _data property is not defined
        '''

        time = dates.localize_datetime(time).astimezone(pytz.utc)
        data_for_element = self._data.get(time, {}).get("element_id", {})

        # TODO: What happens if we have two different flows that both are
        # assigned to the same element? Example: 100 m^3/s @ 1 psu then
        # another flow on the same element of 1 m^3/s @ 100 psu. How do we
        # combine these on a single element? Flow is just simple summation,

        _tmp = data_for_element.get("temperature", np.nan)
        if not np.isnan(_tmp):
            if _tmp != temperature:
                raise NotImplementedError(
                    "Two different values of temperature for same " "time/element."
                )

        _sal = data_for_element.get("salinity", np.nan)
        if not np.isnan(_sal):
            if _sal != salinity:
                raise NotImplementedError(
                    "Two different values of salinity for same time/element."
                )

        self._data.setdefault(time, {}).setdefault(element_id, {}).update(
            {
                "flow": np.nansum([data_for_element.get("flow", np.nan), flow]),
                "temperature": temperature,
                "salinity": salinity,
            }
        )
        if hasattr(self, "_df"):
            del self._df

    def get_element_timeseries(self, element_id):
        data = {}
        element_data = self.df[(self.df["element_id"] == element_id)]
        for row in element_data.sort_values(by=["time"]).itertuples():
            data.setdefault(row.time, {}).update(
                {
                    "flow": row.flow,
                    "temperature": row.temperature,
                    "salinity": row.salinity,
                }
            )
        return data

    def remove_element_timeseries(self, element_id):
        for time in self._data:
            self._data[time].pop(element_id)
        if hasattr(self, "_df"):
            del self._df

    def aggregate_by_radius(self, hgrid, radius):

        logger.info("Begin aggregate_by_radius...")
        start = datetime.now()
        # --- Generate aggregation mapping
        # gather extreme values
        source_max = {element_id: -float("inf")
                      for element_id in self.sources.elements}
        for element_data in self.sources.data.values():
            for element_id, data in element_data.items():
                source_max[element_id] = np.max(
                    [source_max[element_id], data["flow"]])

        sink_max = {element_id: float("inf")
                    for element_id in self.sinks.elements}
        for element_data in self.sinks.data.values():
            for element_id, data in element_data.items():
                sink_max[element_id] = np.min(
                    [sink_max[element_id], data["flow"]])

        aggregate_gdf = []
        for element_id, maxflow in {**source_max, **sink_max}.items():
            element_index = hgrid.elements.get_index_by_id(element_id)
            aggregate_gdf.append(
                {
                    "element_id": element_id,
                    "geometry": hgrid.elements.gdf.iloc[element_index].geometry,
                    "maxflow": maxflow,
                }
            )
        aggregate_gdf = gpd.GeoDataFrame(aggregate_gdf, crs=hgrid.crs).sort_values(
            by="maxflow", key=abs, ascending=False
        )

        aggregation_mapping = {}
        for row in aggregate_gdf.itertuples():
            if row.element_id in aggregation_mapping:
                continue
            aggregation_mapping[row.element_id] = row.element_id
            possible_sources = aggregate_gdf.loc[
                aggregate_gdf.index.difference(
                    np.where(
                        aggregate_gdf["element_id"].isin(
                            list(aggregation_mapping))
                    )
                )
            ]
            circle = get_circle_of_radius(
                row.geometry.centroid.x, row.geometry.centroid.y, radius
            )
            sources_in_circle = possible_sources.loc[possible_sources.within(
                circle)]
            for row_in_circle in sources_in_circle.itertuples():
                aggregation_mapping[row_in_circle.element_id] = row.element_id

        # --- move data from one element to the other
        for current, target in aggregation_mapping.items():
            for time, data in self.get_element_timeseries(current).items():
                self._data[time][target]["flow"] = (
                    self._data[time][current]["flow"] +
                    self._data[time][target]["flow"]
                )

        for current, target in aggregation_mapping.items():
            if current != target:
                self.remove_element_timeseries(current)

        if hasattr(self, "_sources"):
            del self._sources

        if hasattr(self, "_sinks"):
            del self._sinks

        if hasattr(self, "_df"):
            del self._df

        logger.info(f"aggregate_by_radius took {datetime.now()-start}...")

    @staticmethod
    def open(source_sink, vsource, vsink, msource, start_date=None):
        raise NotImplementedError

    def write(
        self,
        path: Union[str, os.PathLike],
        overwrite: bool = False,
        msource: Union[str, bool] = True,
        vsource: Union[str, bool] = True,
        vsink: Union[str, bool] = True,
        source_sink: Union[str, bool] = True,
    ):

        path = pathlib.Path(path)
        path.mkdir(exist_ok=overwrite, parents=True)

        # unpack sources, sinks
        sources, sinks = self.sources, self.sinks

        # write source sink
        if source_sink is True:
            fname = "source_sink.in"
        elif isinstance(source_sink, str):
            fname = source_sink
        if source_sink is not False:
            SourceSinkWriter(sources, sinks, fname).write(path, overwrite)

        if vsource is True:
            fname = "vsource.th"
        elif isinstance(vsource, str):
            fname = vsource
        if vsource is not False:
            Vsource(sources, self.start_date, self.rnday,fname).write(path, overwrite)

        if msource is True:
            fname = "msource.th"
        elif isinstance(msource, str):
            fname = msource
        if msource is not False:
            Msource(sources, self.start_date, self.rnday,fname).write(path, overwrite, )

        if vsink is True:
            fname = "vsink.th"
        elif isinstance(vsink, str):
            fname = vsink
        if vsink is not False:
            Vsink(sinks, self.start_date, self.rnday,fname).write(path, overwrite)

    @property
    def sources(self):
        if not hasattr(self, "_sources"):
            sources = {}
            for element_id in list(
                map(str, list(sorted(map(int, self.df.element_id.unique()))))
            ):
                element_data = self.df[(self.df["element_id"] == element_id)]
                flow_data = element_data["flow"]
                if np.all(flow_data > 0.0):
                    # TODO:  Are irregular timeseries allowed?
                    # if not, we need an interpolator here.
                    for row in element_data.sort_values(by=["time"]).itertuples():
                        sources.setdefault(row.time, {})[element_id] = {
                            "flow": row.flow,
                            "temperature": row.temperature,
                            "salinity": row.salinity,
                        }

                # handle elements that are both sources and sinks
                elif not np.all(flow_data < 0) and np.any(flow_data > 0.0):
                    for row in element_data.sort_values(by=["time"]).itertuples():
                        flow = row.flow if row.flow >= 0.0 else 0.0
                        sources.setdefault(row.time, {})[element_id] = {
                            "flow": flow,
                            "temperature": row.temperature,
                            "salinity": row.salinity,
                        }
            self._sources = Sources(sources)
        return self._sources

    @sources.setter
    def sources(self, sources: Sources = None):
        if sources is not None:
            self._sources = sources

    @property
    def sinks(self):
        if not hasattr(self, "_sinks"):
            sinks = {}
            for element_id in list(
                map(str, list(sorted(map(int, self.df.element_id.unique()))))
            ):
                element_data = self.df[(self.df["element_id"] == element_id)]
                flow_data = element_data["flow"]
                if np.all(flow_data < 0.0):
                    # TODO:  Are irregular timeseries allowed?
                    # if not, we need an interpolator here.
                    for row in element_data.sort_values(by=["time"]).itertuples():
                        sinks.setdefault(row.time, {})[element_id] = {
                            "flow": row.flow}
                # handle elements that are both sources and sinks
                elif not np.all(flow_data > 0.0) and np.any(flow_data < 0.0):
                    for row in element_data.sort_values(by=["time"]).itertuples():
                        sinks.setdefault(row.time, {})[element_id] = {
                            "flow": row.flow if row.flow <= 0.0 else 0.0
                        }
            self._sinks = Sinks(sinks)
        return self._sinks
    
    @sinks.setter
    def sinks(self, sinks: Sinks = None):
        if sinks is not None:
            self._sinks = sinks

    @property
    def start_date(self):
        if not hasattr(self, "_start_date"):
            return self.df.time.min()
        elif self._start_date is None:
            return self.df.time.min()
        return self._start_date

    @start_date.setter
    def start_date(self, start_date):
        self._start_date = start_date
        if start_date is not None:
            self._start_date = dates.localize_datetime(
                start_date).astimezone(pytz.utc)
        return self._start_date

    @property
    def rnday(self):
        if not hasattr(self, "_rnday"):
            return self.df.time.max() - self.df.time.min()
        if self._rnday is None:
            return self.df.time.max() - self.df.time.min()
        return self._rnday

    @rnday.setter
    def rnday(self, rnday):
        self._rnday = rnday
        if rnday is not None:
            self._rnday = (
                rnday if isinstance(
                    rnday, timedelta) else timedelta(days=rnday)
            )
        return self._rnday

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: dict = None):
        # set data and invalidate cached df
        self._data = data or {}
        self._df = None

    @property
    def df(self):
        if self._df is None:
            rows = []
            for time, element_data in self._data.items():
                for element_id, d in element_data.items():
                    rows.append({"time": time, "element_id": element_id, **d})
            self._df = pd.DataFrame(rows)
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame = None):
        self._df = df if df is not None else None      

@lru_cache(maxsize=None)
def get_circle_of_radius(lon, lat, radius):
    wgs84 = CRS.from_user_input("+proj=longlat +datum=WGS84 +no_defs")
    aeqd = CRS.from_user_input(
        "+proj=aeqd +R=6371000 +units=m " f"+lat_0={lat} +lon_0={lon}"
    )
    wgs84_to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    aeqd_to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
    center = Point(float(lon), float(lat))
    point_transformed = ops.transform(wgs84_to_aeqd, center)
    return ops.transform(aeqd_to_wgs84, point_transformed.buffer(radius))


class HGridElementPairings:
    def __init__(self, hgrid: Hgrid, ds: xr.Dataset=None, lon: Union[list,np.array]=None, lat: Union[list, np.array]=None, workers: int = -1, depth_threshold: float = None,buffer:float=0.1):
        """
        Maps (lon, lat) coordinates to the nearest hgrid element ID.
        
        Parameters:
        hgrid : Hgrid object representing the model grid.
        ds : xr dataset with source, time dims/coords and lon, lat, Q, temp, salt data vars
        lon : array-like, Longitudes of the points to be mapped.
        lat : array-like, Latitudes of the points to be mapped.
        workers : int, optional, Number of workers for cKDTree query (default is -1, meaning auto).
        depth_threshold : float (>0), depth threshold to map hgrid source elements to
        buffer : float(>0), buffer distance in units of hgrid to buffer hull of hgrid to for subsetting (lon,lat) source points 
        """

        # hull: shapely Polygon from hgrid
        H = hgrid.hull().iloc[0].geometry.buffer(buffer)
        self._hgrid = hgrid
        if ds is not None:
            points = [Point(x, y) for x, y in zip(ds.lon.values, ds.lat.values)]
            mask = np.fromiter((H.covers(pt) for pt in points), dtype=bool, count=len(points))
            self.ds = ds.isel(source=mask)
            self.source = self.ds.source.values
            self.source_lon=self.ds.lon.values
            self.source_lat=self.ds.lat.values
        elif lon is not None and lat is not None:
            points = [Point(x, y) for x, y in zip(np.asarray(lon), np.asarray(lat))]
            mask = np.fromiter((H.covers(pt) for pt in points), dtype=bool, count=len(points))
            source = np.arange(lon.ravel().shape[0])
            self.ds = None
            self.source = source[mask]
            self.source_lon=lon[mask]
            self.source_lat=lat[mask]

        # Generate element centroids KDTree
        # NOTE! Could use hgrid.elements.compute_centroid() to get lon,lat,depth of element ...
        # ... but this may miss a dry node (and thus element) ... so use for loop
        print('Pairing source elements ... ')

        # get elements in mesh hull
        x_centr, y_centr, dp_centr = hgrid.elements.compute_centroid()
        

        # --- get element id
        # similar to .triangle property for a mesh with all triangles 
        ar = hgrid.elements.array # 0-based index of element
        element_idx = np.arange(ar.shape[0])

        # for element_id, element in hgrid.elements.elements.items(): # --> element id to node id dict map
        #     node_ind = list(map(hgrid.nodes.get_index_by_id, element))
        
        # --- apply depth threshold: 
        # an element is wet if all of its nodes and sides are wet, and is dry if any of its nodes or sides becomes dry.
        # A node/side is wet iff (if and only if) at least 1 of its surrounding element is wet.
        depth_mask = np.all(-hgrid.nodes.values[ar]>depth_threshold,axis=1) 
        centroids = np.column_stack((x_centr[depth_mask], y_centr[depth_mask]))

        self.tree = cKDTree(centroids)
        element_idx = element_idx[depth_mask]
        element_id = np.array(element_idx + 1,dtype=str)
        element_centroid_lon=x_centr[depth_mask]
        element_centroid_lat=y_centr[depth_mask]
        element_mean_depth=dp_centr[depth_mask]

        # Find nearest elements for given lon/lat points
        coords = np.vstack((self.source_lon, self.source_lat)).T
        _, idxs = self.tree.query(coords, workers=workers) # idxs = linear index to masked element centroid lon/lat corresponding to self.source

        assert len(idxs) == len(self.source)

        # self._idxs = idxs # note: index to map lon, lat to element centroids
        # self.map = {'source':self.ds.source.values, 'hgrid_element_id':self.element_ids[idxs]}
        self.map = dict(zip(self.ds.source.values, zip(element_idx[idxs], element_id[idxs])))
        self.element_idx= element_idx[idxs] 
        self.element_id= element_id[idxs] 
        self.element_centroid_lon = element_centroid_lon[idxs]
        self.element_centroid_lat = element_centroid_lat[idxs]
        self.element_mean_depth = element_mean_depth[idxs]

        # # debug

        # centroids = []
        # element_ids = []
        # element_centroid_lon=[]
        # element_centroid_lat=[]
        # element_mean_depth=[]
        # for element_id, element_nodes in hgrid.elements.elements.items():
        #     #node_idx = hgrid.nodes.get_index_by_id(element_nodes)
        #     node_idx = list(map(hgrid.nodes.get_index_by_id, element_nodes))
        #     # apply depth threshold: 
        #     # an element is wet if all of its nodes and sides are wet, and is dry if any of its nodes or sides becomes dry.
        #     # A node/side is wet iff (if and only if) at least 1 of its surrounding element is wet.
        #     node_depths = -hgrid.nodes.values[node_idx] # -1 * hgrid.node.values s.t. (z, negative down) --> (depth, positive down)
        #     if np.any(node_depths < depth_threshold): # dont use this, it is redundant: or not np.all(H.covers([Point(x, y) for x, y in zip(hgrid.nodes.coords[node_ind,0], hgrid.nodes.coords[node_ind,1])]):  
        #         # do not consider element that in cKDTree nn search
        #         continue
        #     centroid = LineString(hgrid.nodes.coord[node_idx]).centroid
        #     centroids.append((centroid.x, centroid.y))
        #     element_ids.append(element_id)
        #     element_centroid_lon.append(centroid.x)
        #     element_centroid_lat.append(centroid.y)    
        #     element_mean_depth.append(np.mean(node_depths))    
        
        
        # element_ids = np.array(element_ids)
        # element_centroid_lon=np.array(element_centroid_lon)
        # element_centroid_lat=np.array(element_centroid_lat)
        # element_mean_depth=np.array(element_mean_depth)
    
        # tree = cKDTree(centroids)
        # coords = np.vstack((self.source_lon, self.source_lat)).T
        # _, idxs = tree.query(coords, workers=workers) # idxs = linear index to masked element centroid lon/lat corresponding to self.source
        # assert len(idxs) == len(self.source)
        # self.bmap = dict(zip(self.ds.source.values, zip(np.array(element_ids[idxs],dtype=int)-1,element_ids[idxs])))
        # self.belement_id=element_ids[idxs] 
        # self.belement_centroid_lon = element_centroid_lon[idxs]
        # self.belement_centroid_lat = element_centroid_lat[idxs]
        # self.belement_mean_depth = element_mean_depth[idxs]

    def get_mapped_elements(self):
        """Returns a dictionary mapping (lon, lat) tuples to element IDs."""
        return {elem_id: (lon, lat, depth) for (lon, lat, depth), elem_id in zip(zip(self.element_centroid_lon, self.element_centroid_lat,self.element_mean_depth), self.element_id)}
    
    def save_json(self, filename):
        """Saves the mappings to a JSON file."""
        import json
        with open(filename, "w") as f:
            json.dump(self.get_mapped_elements(), f)

    def to_dict(self, start_date: datetime, rnday:timedelta, as_climatology=True):
        
        start_date = dates.localize_datetime(start_date).astimezone(pytz.utc)

        if "temp" not in self.ds:
            self.ds["temp"] = xr.full_like(self.ds["Q"], -9999.0).assign_attrs(
                long_name="temperature", units="degC"
            )

        if "salt" not in self.ds:
            self.ds["salt"] = xr.full_like(self.ds["Q"], 0).assign_attrs(
                long_name="salinity", units="psu"
            )
        if as_climatology:
            # Loop over time steps 
            print('ds to Sources ... ')
            data = {}
            for i, dt in enumerate( [start_date + timedelta(days=i) for i in range(rnday.days+2)] ):
                day_of_year = int(dt.strftime('%j'))
                q_data = self.ds['Q'].sel(day=day_of_year).values
                temp_data = self.ds['temp'].sel(day=day_of_year).values
                salt_data = self.ds['salt'].sel(day=day_of_year).values

                element_data = {}
                for i in self.map.keys():
                    eid = self.map[i][1]
                    assert isinstance(eid,str)
                    element_data[eid] = {
                            "flow": q_data[i],  # Direct access
                            "temperature": temp_data[i], # ambient value in model = -9999.0
                            "salinity": salt_data[i] # psu, ambient value in model = -9999.0
                        }

                data[dt] = element_data
                
        return data

    def to_SourceSink(self, start_date: datetime, rnday:timedelta, as_climatology=True):

        if "temp" not in self.ds:
            self.ds["temp"] = xr.full_like(self.ds["Q"], -9999.0).assign_attrs(
                long_name="temperature", units="degC"
            )

        if "salt" not in self.ds:
            self.ds["salt"] = xr.full_like(self.ds["Q"], 0).assign_attrs(
                long_name="salinity", units="psu"
            )
        if as_climatology:
            ss = SourceSink()
            for i, dt in enumerate( [start_date + timedelta(days=i) for i in range(rnday.days+2)] ):
                day_of_year = int(dt.strftime('%j'))
                flow_data = self.ds['Q'].sel(day=day_of_year).values
                temp_data = self.ds['temp'].sel(day=day_of_year).values
                salt_data = self.ds['salt'].sel(day=day_of_year).values

                element_data = {}
                for i in self.map.keys():
                    eid = self.map[i][1]
                    assert isinstance(eid,str)
                    ss.add_data(
                        time = dt,
                        element_id=eid,
                        flow=flow_data[i], # m3/s, positive=flow entering, negative=flow leaving
                        temperature=temp_data[i],   # ambient value in model = -9999.0
                        salinity=salt_data[i]       # psu, ambient value in model = -9999.0
                    )
            return ss
    
    def make_plot(self,ax = None,figsize=(10,10),show_text_labels=False):
        """Plots the mapped points and hgrid elements for verification."""
       
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.hgrid.triplot(axes=ax,label="hgrid")

        gdf = self.hgrid.elements.gdf
        for id in self.element_id: gdf[gdf.id == id].geometry.plot(ax=ax)

        ax.scatter(self.source_lon, self.source_lat, marker=".", color="cyan", label="Original Source Points")
        cs = ax.scatter(
            self.element_centroid_lon,
            self.element_centroid_lat,
            s=5,
            c=np.asarray(self.element_mean_depth, dtype=float),
            marker="o",
            label="Mapped Element Centroid to Source Points",
        )
        fig.colorbar(cs, ax=ax, label="Mean Element Depth")
        # Label elements
        if show_text_labels:
            for x, y, id in zip(self.element_centroid_lon, self.element_centroid_lat, self.element_id):
                ax.text(x, y, id, fontsize=6, ha="center", va="top")
            # for element_id, element in self.hgrid.elements.elements.items():
            #     node_ind = list(map(self.hgrid.nodes.get_index_by_id, element))
            #     x = np.mean(self.hgrid.coords[node_ind,0])
            #     y = np.mean(self.hgrid.coords[node_ind,1])
            #     ax.text(x, y, str(element_id), fontsize=6, ha="center", va="center",color='r')

        # Draw connectors from each source point to its mapped centroid
        for x0, y0, x1, y1 in zip(self.source_lon, self.source_lat, self.element_centroid_lon, self.element_centroid_lat):
            ax.plot([x0, x1], [y0, y1], linestyle="-", color="lightgrey",linewidth=0.8, alpha=0.7)

        # # debug
        # if debug:
        #     ax.scatter(self.source_lon, self.source_lat, marker=".", color="cyan", label="Original Source Points")
        #     cs = ax.scatter(
        #         self.belement_centroid_lon,
        #         self.belement_centroid_lat,
        #         s=5,
        #         c=np.asarray(self.element_mean_depth, dtype=float),
        #         marker="s",
        #         label="db Mapped Element Centroid to Source Points",
        #     )
        #     # Label elements
        #     if show_text_labels:
        #         for x, y, id in zip(self.belement_centroid_lon, self.belement_centroid_lat, self.belement_id):
        #             ax.text(x, y, id, fontsize=6, ha="center", va="bottom", color='r')
        
        #     # for element_id, element_node_id in self.hgrid.elements.elements.items():
        #     #     node_ind = list(map(self.hgrid.nodes.get_index_by_id, element_node_id))
        #     #     x = np.mean(self.hgrid.coords[node_ind,0])
        #     #     y = np.mean(self.hgrid.coords[node_ind,1])
        #     #     ax.text(x, y, str(element_id), fontsize=4, ha="center", va="center",color='b')

        # # Draw connectors from each source point to its mapped centroid
        # for x0, y0, x1, y1 in zip(self.source_lon, self.source_lat, self.belement_centroid_lon, self.belement_centroid_lat):
        #     ax.plot([x0, x1], [y0, y1], linestyle="-", color="darkgrey",linewidth=0.8, alpha=0.7)
        # # end debug

        if not show_text_labels:
            plt.legend()
        bbox=self.hgrid.bbox
        minx,miny = bbox.min
        maxx,maxy = bbox.max
        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_aspect("equal", adjustable="box")
        plt.show()

    @property
    def hgrid(self):
        return self._hgrid
