import pathlib, os
import matplotlib.animation as mpl_animation
# from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from netCDF4 import Dataset, MFDataset, num2date # consider replacing with xarray ... 
import pandas as pd
import xarray as xr
import numpy as np
from pyugrid import UGrid
import contextily
from pyschism.enums import OutputVariableUnit, OutputVariableShortName


class PlotOutputCombined:

    def __init__(self, path):
        
        if isinstance(path,list):
            self.parent = pathlib.Path(path[0]).parent
            self.files = path
            self.name = self.files[0].name
            self.ds = xr.open_mfdataset(self.files)
    
        elif isinstance(path,os.PathLike):
            self.parent = pathlib.Path(path).parent
            self.files = pathlib.Path(path)
            self.name = self.files.name
            self.ds = xr.open_dataset(self.files)

        # tvar = self.nc.variables["time"]
        # units = tvar.units                               # "seconds since YYYY-MM-DDThh:mm:ss+00"
        # calendar = getattr(tvar, "calendar", "standard") # default if missing

        # # set datetimes (python or cftime as needed)
        # self.time = num2date(tvar[:], units=units, calendar=calendar,
        #                 only_use_cftime_datetimes=True,
        #                 only_use_python_datetimes=False)

        # # # Convenient: make a tz-aware pandas index in UTC
        # # self.time = pd.DatetimeIndex(t_py).tz_localize("UTC")

    def plot(self, 
            variable, 
            index=None,
            unit=None,
            show=False, 
            wireframe=False,
            figsize=None, 
            cmap='jet', 
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            vmin=None,
            vmax=None,
            levels=256,
            cax=None, # optional args if called from PlotOutputCombined.animation.animate method
            fig=None,
            ax=None,
            triangulation=None
             ):
        
        # --- define mesh triangulation
        if triangulation is None:
            # ugrid = UGrid.from_nc_dataset(self.nc)
            # x = ugrid.nodes[:, 0]
            # y = ugrid.nodes[:, 1]
            # triangulation = Triangulation(x, y, ugrid.faces[:, :3])

            # Face connectivity (assume 1-based indexing)
            x,y=self.ds.SCHISM_hgrid_node_x.values,self.ds.SCHISM_hgrid_node_y.values
            tri = self.ds['SCHISM_hgrid_face_nodes'].isel(time=index).values[:, :3] - 1
            triangulation = Triangulation(x, y, tri)

        # --- get output var at index
        print(f'{variable}: time index={index} of {self.ds.time.shape[0]}')
        val = self.ds[variable].isel(time=index).values

        # --- mask
        if 'wetdry_elem' in self.ds.data_vars: 
            triangulation.set_mask(self.ds['wetdry_elem'].isel(time=index).values)
        elif 'dryFlagNode' in self.ds.data_vars: 
            # mask triangles that touch any non-finite node value
            tri_masks = []
            mask = self.ds['dryFlagNode'].isel(time=index) == 1
            val[mask] = np.nan
            nodes_finite = np.isfinite(val)
            bad_tri = ~nodes_finite[triangulation.triangles].all(axis=1)
            tri_masks.append(bad_tri)
            tri_mask = np.logical_or.reduce(tri_masks)
            triangulation.set_mask(tri_mask)
            val=np.ma.array(val, mask=~np.isfinite(val))
        else:    
            print(f'no mask applied in PlotOutputCombined.animation, index={index}')
        

        # --- plot
        if fig is None and ax is None:
            if figsize is None:
                fig,ax = plt.subplots(layout="constrained")
                mng = plt.get_current_fig_manager()
                # Qt5/Qt6Agg
                if hasattr(mng, "window"):
                    mng.window.showMaximized()      # or: mng.window.showFullScreen()

                # TkAgg
                if hasattr(mng, "window") and hasattr(mng.window, "state"):
                    mng.window.state("zoomed")      # or: mng.window.attributes("-fullscreen", True)

                # WXAgg
                if hasattr(mng, "frame"):
                    mng.frame.Maximize(True)
            else:
                fig,ax = plt.subplots(figsize=figsize,layout="constrained")
            # plt.tight_layout(pad=2)

        ax.tricontourf(
            triangulation,
            val,
            cmap=cmap,
            levels=levels,
            vmin=vmin,
            vmax=vmax
            )

        if wireframe:
            ax.triplot(triangulation, color='k', linewidth=0.05)

        ax.set_aspect('equal')
        # ax.axis('scaled')
        ax.set_ylim(ymin, ymax, auto=True)
        ax.set_xlim(xmin, xmax, auto=True)
        
        ax.set_xlabel('Longitude [deg E]')
        ax.set_ylabel('Latitude [deg N]')

        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(val)
        m.set_clim(vmin, vmax)
        if cax is None:
            cbar = fig.colorbar(m, ax=ax, format='%.1f', boundaries=np.linspace(vmin, vmax, levels), pad=0.02, shrink=0.5)
        else:
            cbar = fig.colorbar(m, cax=cax, format='%.1f', boundaries=np.linspace(vmin, vmax, levels), pad=0.02, shrink=0.5)
        cbar.ax.set_ylabel(f'{variable} [{unit}]', rotation=90)

        # Format the datetime as a string (drop miliseconds)
        timestamp = np.datetime_as_string(self.ds['time'].isel(time=index).values, unit='m')
        ax.set_title(timestamp)
        if show:
            plt.show()

    def animation(
            self,
            variable,
            unit='',
            save=False,
            savedir=pathlib.Path('./'),
            fileformat='mp4',
            fps=3,
            dpi=150,
            start_frame=0,
            end_frame=-1,
            figsize=None,
            wireframe=False,
            cmap='jet',
            levels=256,
            show=False,
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            vmin=None,
            vmax=None,
            add_basemap=False,
            ctx={'crs':'EPSG:4326','source':contextily.providers.Esri.WorldImagery,'zoom':'auto'}
    ):

        if figsize is None:
            fig,ax = plt.subplots(layout="constrained")
            mng = plt.get_current_fig_manager()
            # Qt5/Qt6Agg
            if hasattr(mng, "window"):
                mng.window.showMaximized()      # or: mng.window.showFullScreen()

            # TkAgg
            if hasattr(mng, "window") and hasattr(mng.window, "state"):
                mng.window.state("zoomed")      # or: mng.window.attributes("-fullscreen", True)

            # WXAgg
            if hasattr(mng, "frame"):
                mng.frame.Maximize(True)
        else:
            fig,ax = plt.subplots(figsize=figsize,layout="constrained")

        if add_basemap:
            # | Name                           | Code                              |
            # | ------------------------------ | --------------------------------- |
            # | Esri World Imagery (Satellite) | `ctx.providers.Esri.WorldImagery` |
            # | OpenStreetMap                | `ctx.providers.OpenStreetMap.Mapnik` |
            # | Esri World Street Map        | `ctx.providers.Esri.WorldStreetMap`  |
            # | OpenTopoMap                  | `ctx.providers.OpenTopoMap`          |
            # | Stamen Terrain               | `ctx.providers.Stamen.Terrain`       |
            # | Stamen Toner (high contrast) | `ctx.providers.Stamen.Toner`         |
            # | Stamen Watercolor                     | `ctx.providers.Stamen.Watercolor`  |
            # | CartoDB Positron (light gray)         | `ctx.providers.CartoDB.Positron`   |
            # | CartoDB Dark Matter (dark background) | `ctx.providers.CartoDB.DarkMatter` |
            contextily.add_basemap(ax, crs=ctx['crs'], source=ctx['source'], zoom=ctx['zoom'])

        
        # plt.tight_layout(pad=2)
        # plt.tight_layout(pad=2)

        # ugrid = UGrid.from_nc_dataset(self.nc)
        # x = ugrid.nodes[:, 0]
        # y = ugrid.nodes[:, 1]
        # triangulation = Triangulation(x, y, ugrid.faces[:, :3])

        # Face connectivity (assume 1-based indexing)
        x,y=self.ds.SCHISM_hgrid_node_x.values,self.ds.SCHISM_hgrid_node_y.values
        tri = self.ds['SCHISM_hgrid_face_nodes'].isel(time=0).values[:, :3] - 1
        triangulation = Triangulation(x, y, tri)

        xmin = np.min(x) if xmin is None else xmin
        xmax = np.max(x) if xmax is None else xmax
        ymin = np.min(y) if ymin is None else ymin
        ymax = np.max(y) if ymax is None else ymax
        vmin = np.min(self.self[variable]) if vmin is None else vmin
        vmax = np.max(self.self[variable]) if vmax is None else vmax
        # unit = OutputVariableUnit[OutputVariableShortName(variable).name].value
        # unit = ''

        def animate(index):
            _ax = fig.get_axes()
            ax.clear()
            if len(_ax) > 1:
                cax = _ax[1]
                cax.cla()
            else:
                cax = None

            self.plot(
                variable=variable,
                index=index,
                unit=unit,
                show=show,
                wireframe=wireframe,
                cmap=cmap,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                vmin=vmin,
                vmax=vmax,
                levels=levels,
                cax=cax,
                fig=fig,
                ax=ax,
                triangulation=triangulation
            )

        end_frame = end_frame % self.ds[variable].shape[0] \
            if end_frame < 0 else end_frame
        start_frame = start_frame % self.ds[variable].shape[0] \
            if start_frame < 0 else start_frame
        frames = range(start_frame, end_frame)
        anim = mpl_animation.FuncAnimation(
            fig,
            animate,
            frames,
            blit=False
            )

        if save:

            # anim.save(
            #     pathlib.Path(save),
            #     writer='ffmpeg',
            #     fps=fps
            # )

            # (Optional) If ffmpeg isn't on PATH, point to it:
            # mpl.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'

            Writer = mpl_animation.FFMpegWriter
            writer = Writer(fps=fps, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
            start_datetime=np.datetime_as_string(self.ds.time.isel(time=1).values, unit='m').replace(' ','_')
            end_datetime=np.datetime_as_string(self.ds.time.isel(time=-1).values, unit='m').replace(' ','_')
            outfile=savedir / pathlib.Path(f'{self.name}_{variable}_{start_datetime}_{end_datetime}.{fileformat}')
            anim.save(
                outfile,
                writer=writer,
                dpi=dpi
            )
            print('Animation saved to: ', outfile'))
            

        if show:
            plt.show()

        return anim
