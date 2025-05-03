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
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, MultiPoint
from collections import defaultdict
from pyproj import CRS, Transformer
import pytz
from scipy.interpolate import interp1d
from shapely import ops
from shapely.geometry import Point

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
            # self._elements = list(
            #     map(str, sorted(list(map(int, unique_elements)))))
            self._elements = list(
                map(int, sorted(list(map(int, unique_elements)))))
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
            relative_time = (
                self.dataset.timevector[i] - self.start_date).total_seconds()
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

        if isinstance(element_id, str):
            try:
                element_id = int(element_id)  # Convert to integer
            except ValueError:
                raise ValueError(f"Cannot convert '{element_id}' to an integer.")
            
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
            self.sources = sources  
        # else:
        #     self.sources = Sources()

        if sinks is not None:
            self.sinks = sinks  
        # else:
        #     self.sinks()

        if sinks is not None:
            self._data = data  
        # else:
        #     self._data = {}


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
            Vsource(sources, self.start_date, self.rnday,
                    fname).write(path, overwrite)

        if msource is True:
            fname = "msource.th"
        elif isinstance(msource, str):
            fname = msource
        if msource is not False:
            Msource(sources, self.start_date, self.rnday,
                    fname).write(
                        path, 
                        overwrite,
                        )

        if vsink is True:
            fname = "vsink.th"
        elif isinstance(vsink, str):
            fname = vsink
        if vsink is not False:
            Vsink(sinks, self.start_date, self.rnday,
                  fname).write(path, overwrite)

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
        if data is not None:
            self._data = data

    @property
    def df(self):
        if not hasattr(self, "_df"):
            _data = []
            for time, element_data in self._data.items():
                for element_id, data in element_data.items():
                    _data.append(
                        {"time": time, "element_id": element_id, **data})
            self._df = pd.DataFrame(_data)
        return self._df
    
    @data.setter
    def df(self, df: pd.DataFrame = None):
        if df is not None:
            self._df = df        

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
    def __init__(self, hgrid: Union[Hgrid, Gr3], lon: Union[list,np.array], lat: Union[list, np.array], workers: int = -1, depth_threshold: float = None):
        """
        Maps (lon, lat) coordinates to the nearest hgrid element ID.
        
        Parameters:
        hgrid : Gr3, the hgrid object representing the model grid.
        lon : array-like, Longitudes of the points to be mapped.
        lat : array-like, Latitudes of the points to be mapped.
        workers : int, optional, Number of workers for cKDTree query (default is -1, meaning auto).
        """
        self._hgrid = hgrid
        self.lon = np.asarray(lon)
        self.lat = np.asarray(lat)
        
        # Generate element centroids KDTree
        centroids = []
        element_ids = []
        for element_id, element in hgrid.elements.elements.items():
            node_ind = list(map(hgrid.nodes.get_index_by_id, element))

            # apply depth threshold: 
            # an element is wet if all of its nodes and sides are wet, and is dry if any of its nodes or sides becomes dry.
            # A node/side is wet iff (if and only if) at least 1 of its surrounding element is wet.
            if depth_threshold is not None:
                if any(hgrid.nodes.values[node_ind] > depth_threshold):  
                    continue 
            centroid = LineString(
                hgrid.nodes.coord[node_ind]
            ).centroid
            centroids.append((centroid.x, centroid.y))
            element_ids.append(element_id)
        
        self.tree = cKDTree(centroids)
        self.element_ids = np.array(element_ids)
        
        # Find nearest elements for given lon/lat points
        # map each (lon, lat) point to the nearest element using KDTree."""
        coords = np.vstack((self.lon, self.lat)).T
        _, idxs = self.tree.query(coords, workers=workers)
        self.idxs = idxs
        self.mapped_elements = self.element_ids[idxs]
    
    def get_mapped_elements(self):
        """Returns a dictionary mapping (lon, lat) tuples to element IDs."""
        return {(lon, lat): elem_id for (lon, lat), elem_id in zip(zip(self.lon, self.lat), self.mapped_elements)}
    
    def save_json(self, filename):
        """Saves the mappings to a JSON file."""
        import json
        with open(filename, "w") as f:
            json.dump(self.get_mapped_elements(), f)
    
    def make_plot(self,ax = None,figsize=(10,10)):
        """Plots the mapped points and hgrid elements for verification."""
        gdf = self.hgrid.elements.gdf
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        self.hgrid.triplot(axes=ax,label="hgrid")
        ax.scatter(self.lon, self.lat, marker="x", color="red", label="Mapped Points")
        plt.legend()
        plt.show()

    @property
    def hgrid(self):
        return self._hgrid
