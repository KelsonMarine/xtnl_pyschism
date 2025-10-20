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
import re

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

            
        # Extract variable names that end with X or Y
        vars_xy = [var for var in self.ds.data_vars if re.search(r'(X|Y)$', var)]

        # Group by base name (everything before last char)
        groups = {}
        for var in vars_xy:
            base = var[:-1]  # remove trailing X or Y
            groups.setdefault(base, []).append(var)

        # Find groups that have both X and Y
        for base, vars_list in groups.items():
            if len(vars_list) == 2 and f"{base}X" in vars_list and f"{base}Y" in vars_list:
                vx = self.ds[f"{base}X"]
                vy = self.ds[f"{base}Y"]
                mag = np.sqrt(vx**2 + vy**2)
                self.ds[f"{base}Mag"] = mag        

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
            triangulation=None,
            show_cbar=True
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

        if show_cbar:
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(val)
            m.set_clim(vmin, vmax)
            if cax is None:
                cbar = fig.colorbar(m, ax=ax, format='%.2f', boundaries=np.linspace(vmin, vmax, levels), pad=0.02, shrink=0.5)
            else:
                cbar = fig.colorbar(m, cax=cax, format='%.2f', boundaries=np.linspace(vmin, vmax, levels), pad=0.02, shrink=0.5)
            label = variable if not unit else f"{variable} [{unit}]"
            cbar.set_label(label)

        ax.set_title(f"{variable} | {np.datetime_as_string(self.ds.time.isel(time=index).values, unit='m')}")
        if show:
            plt.show()

    def animation(
            self,
            variable,
            unit='',
            save=False,
            savedir=pathlib.Path('./'),
            fname=None,
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
            ctx_opts={'crs':'EPSG:4326','source':contextily.providers.Esri.WorldImagery,'zoom':'auto'},
            show_vector_field=False,
            show_cbar=True,
        ):

        # -------------------------
        # Figure / window behavior
        # -------------------------
        if figsize is None:
            fig, ax = plt.subplots(layout="constrained")
            mng = plt.get_current_fig_manager()
            if hasattr(mng, "window"):      # Qt
                try: mng.window.showMaximized()
                except Exception: pass
            if hasattr(mng, "window") and hasattr(mng.window, "state"):  # Tk
                try: mng.window.state("zoomed")
                except Exception: pass
            if hasattr(mng, "frame"):       # WX
                try: mng.frame.Maximize(True)
                except Exception: pass
        else:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        # Face connectivity (1-based indexing used in netcdf)
        x = self.ds.SCHISM_hgrid_node_x.values  # lon or x
        y = self.ds.SCHISM_hgrid_node_y.values  # lat or y
        if 'time' in self.ds['SCHISM_hgrid_face_nodes'].coords:
            tri = self.ds['SCHISM_hgrid_face_nodes'].isel(time=0).values[:, :3] - 1
        else:
            tri = self.ds['SCHISM_hgrid_face_nodes'].values[:, :3] - 1

        # value range (finite only)
        if vmin is None or vmax is None:
            vals_all = np.asarray(self.ds[variable]).ravel()
            finite_vals = vals_all[np.isfinite(vals_all)]
            if finite_vals.size:
                if vmin is None: vmin = float(np.nanmin(finite_vals))
                if vmax is None: vmax = float(np.nanmax(finite_vals))
        triangulation = Triangulation(x, y, tri)
        xmin_plot = np.min(x) if xmin is None else xmin
        xmax_plot = np.max(x) if xmax is None else xmax
        ymin_plot = np.min(y) if ymin is None else ymin
        ymax_plot = np.max(y) if ymax is None else ymax
        ax.set_xlim(xmin_plot, xmax_plot)
        ax.set_ylim(ymin_plot, ymax_plot)
        ax.set_xlabel('Longitude [deg E]')
        ax.set_ylabel('Latitude [deg N]')
        if add_basemap:
            print('adding basemap')
            contextily.add_basemap(ax, crs=ctx_opts['crs'], source=ctx_opts['source'], zoom=ctx_opts['zoom'])
        ax.set_aspect('equal')

        if wireframe:
            ax.triplot(triangulation, color='k', linewidth=0.5, zorder=5)

        # -------------------------
        # First frame: draw once, cache artists, keep colorbar
        # -------------------------
        contourf = None
        cbar = None
        quiver = None
     
        def draw_vector_field(index,first=False):
            nonlocal quiver
            step=10
            u = self.ds['depthAverageVelX'].isel(time=index).values[::step]
            v = self.ds['depthAverageVelY'].isel(time=index).values[::step]
            mag = np.hypot(u,v)
            m = np.isfinite(u) & np.isfinite(v) & (np.hypot(u,v) < vmax)
            u = np.where(m, u, 0.0); 
            v = np.where(m, v, 0.0)

            # normalize to directions
            ux = np.divide(u, mag, out=np.zeros_like(u), where=mag>0)
            vy = np.divide(v, mag, out=np.zeros_like(v), where=mag>0)

            L = 5.0  # arrow length in pixels (points)
            u = ux * L
            v = vy * L
            if first:        
                quiver = ax.quiver(x[::step], y[::step],u,v,
                    color='k',
                    zorder=20,
                    animated=True,
                    angles='xy',
                    scale_units='dots', # pixels
                    scale=1,
                    pivot='tail', # 'tail', 'mid', 'middle', 'tip'
                    width=0.002, # shaft width of arrow
                    headwidth=1, # relative to width
                    headlength=1.5 # relative to width
                )
            else:                
                quiver.set_UVC(u, v)  # update arrow directions/mags
            return quiver              # return for blit=True

        def draw_frame(index, first=False):
            nonlocal contourf, cbar, quiver

            # Get data for this index
            val = np.asarray(self.ds[variable].isel(time=index)).astype(float)

            # (Optional) mask logic if you have wet/dry flags; otherwise ensure no infs            
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
                # val=np.ma.array(val, mask=~np.isfinite(val))
            else:    
                print(f'no mask applied in PlotOutputCombined.animation, index={index}')
            
            # Remove previous filled contour but DO NOT clear axes (keeps basemap)
            if contourf is not None:
                for coll in contourf.collections:
                    coll.remove()
                contourf = None

            # Draw filled contour
            contourf = ax.tricontourf(
                triangulation,
                val,
                levels=levels,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                zorder=10
            )

            if variable == 'depthAverageVelMag' and 'depthAverageVelX' in self.ds and 'depthAverageVelY' in self.ds and show_vector_field:
                quiver=draw_vector_field(index,first=first)

            # Colorbar: create once, then just update its mappable
            if first and show_cbar:
                cbar = fig.colorbar(contourf, ax=ax, format='%.2f', boundaries=np.linspace(vmin, vmax, levels), pad=0.02, shrink=0.5)
                label = variable if not unit else f"{variable} [{unit}]"
                cbar.set_label(label)

            ax.set_title(f"{variable} | {np.datetime_as_string(self.ds.time.isel(time=index).values, unit='m')}")

            return contourf.collections

        # Determine frames
        nT = self.ds['time'].shape[0]
        end_frame = (end_frame % nT) if end_frame < 0 else min(end_frame, nT)
        start_frame = (start_frame % nT) if start_frame < 0 else max(start_frame, 0)

        from tqdm import tqdm
        frames = range(start_frame, end_frame)

        # Draw first frame before FuncAnimation so everything exists
        artists0 = draw_frame(start_frame, first=True)

        def update(index):
            return draw_frame(index, first=False)

        anim = mpl_animation.FuncAnimation(
            fig, update, frames=tqdm(frames, total=len(frames)), blit=False, interval=int(1000/max(fps,1))
        )

        if save:
            # mpl.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg' # may need to define this for python env.
            if not mpl_animation.FFMpegWriter.isAvailable():
                raise("FFMpegWriter not available -- check install and set mpl.rcParams['animation.ffmpeg_path']")
            
            Writer = mpl_animation.FFMpegWriter
            writer = Writer(fps=fps, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

            start_datetime=np.datetime_as_string(self.ds.time.isel(time=1).values, unit='m').replace(' ','_')
            end_datetime=np.datetime_as_string(self.ds.time.isel(time=-1).values, unit='m').replace(' ','_')
            if fname is None:
                fname = pathlib.Path(f'{self.name}_{variable}_{start_datetime}_{end_datetime}.{fileformat}')
            if not isinstance(savedir,pathlib.Path):
                savedir = pathlib.Path(savedir)
            outfile=savedir / fname
            anim.save(
                outfile,
                writer=writer,
                dpi=dpi
            )
            print('Animation saved to: ', outfile)
            

        if show:
            plt.show()

        return anim
