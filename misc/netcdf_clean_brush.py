import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
from netCDF4 import Dataset
import matplotlib.colors as colors

AVOGADRO = 6.02214076e23  # mol^-1
MOLEC_PER_DOBSON_UNIT = 2.687e20   # molecules m^-2 in 1 Dobson unit

#python clean_files.py /path/to/folder --field your_variable
#python clean_files.py <folder> --field <var>

class NetCDFViewer:
    def __init__(self, folder_path, field_name=None):

        self.folder_path = folder_path
        self.nc_files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))

        print (self.nc_files)
        #exit()

        if not self.nc_files:
            raise FileNotFoundError("No NetCDF files found in the specified folder")

        # Initialize current file and indices
        self.current_file_idx = 0
        self.current_ds = None
        # Prefer provided field name; fall back to older ones if needed
        self.current_var_name = field_name or 'aerosol_index_354_388'

        # Initialize eraser properties
        self.eraser_radius = 5
        self.eraser_active = False
        self.eraser_circle = None
        self.modified = False
        # Data / units
        self.data_native = None  # original units, 2D slice
        self.data_units = None
        self.display_scale = 1.0
        self.display_units = None

        # Geographic coordinates
        self.lats = None
        self.lons = None

        # Load first file
        self.load_current_file()

        # Set up the figure and axes
        self.setup_plot()

    def load_current_file(self):
        """Load the current NetCDF file and prepare for display"""
        file_path = self.nc_files[self.current_file_idx]
        
        self.current_ds = Dataset(file_path, 'r')
        self.lats = self.current_ds.variables['lat'][:]
        self.lons = self.current_ds.variables['lon'][:]

        # Try preferred and fallback variable names
        candidate_vars = [
            self.current_var_name,
            'sulfurdioxide_total_vertical_column_raw',
            'aerosol_index_354_388',
            'sulfurdioxide_total_vertical_column_15km',
        ]
        var_name = None
        for name in candidate_vars:
            if name in self.current_ds.variables:
                var_name = name
                break
        if var_name is None:
            self.current_ds.close()
            raise KeyError(
                f"None of the expected variables found in "
                f"{os.path.basename(file_path)}: {candidate_vars}"
            )

        self.current_var_name = var_name
        var = self.current_ds.variables[self.current_var_name]

        # Load first time slice if present, otherwise full 2D field
        if var.ndim == 3:
            data_native = var[0, :, :]
        else:
            data_native = var[:, :]

        self.data_native = np.array(data_native, dtype=float)
        self.data_units = getattr(var, 'units', None)
        self.display_scale, self.display_units = self._compute_display_scale(
            self.data_units
        )

        self.current_ds.close()
        self.modified = False

    def draw_map_features(self):
        """Add background map features and gridlines"""
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        self.ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        gl = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    def setup_plot(self):
        """Set up the matplotlib figure and widgets"""
        #plt.rcParams['toolbar'] = 'None'  # Hide default toolbar

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        self.fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95, top=0.9)
        
        # Add colorbar axes (vertical bar on the right)
        self.cax = self.fig.add_axes([0.92, 0.2, 0.015, 0.6])

        # Add axes for widgets
        self.eraser_button_ax = self.fig.add_axes([0.1, 0.01, 0.15, 0.03])
        self.save_button_ax = self.fig.add_axes([0.3, 0.01, 0.15, 0.03])
        self.next_button_ax = self.fig.add_axes([0.5, 0.01, 0.15, 0.03])
        self.radius_slider_ax = self.fig.add_axes([0.75, 0.01, 0.2, 0.03])

        # Create buttons
        self.eraser_button = Button(self.eraser_button_ax, 'Eraser: Off')
        self.save_button = Button(self.save_button_ax, 'Save to file')
        self.next_button = Button(self.next_button_ax, 'Next file')

        # Create eraser radius slider
        self.radius_slider = Slider(
            self.radius_slider_ax,
            'Eraser Size',
            1,
            100,
            valinit=self.eraser_radius,
            valstep=1,
        )

        # Connect events
        self.eraser_button.on_clicked(self.toggle_eraser)
        self.save_button.on_clicked(self.save_data)
        self.next_button.on_clicked(self.next_file)
        self.radius_slider.on_changed(self.update_eraser_radius)

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # Draw map features initially
        self.draw_map_features()

        # Display initial image
        self.update_plot()

        self.fig.suptitle(os.path.basename(self.nc_files[self.current_file_idx]), fontsize=12)
        plt.show()

    def _compute_display_scale(self, units):
        """Return multiplicative factor to convert from native units to Dobson units."""
        if units is None:
            return 1.0, None

        u = units.lower().strip()
        if "dobson" in u:
            return 1.0, "Dobson units"

        # mol/m2 -> Dobson units
        if "mol" in u and ("m-2" in u or "m^-2" in u or "mol/m2" in u):
            factor = AVOGADRO / MOLEC_PER_DOBSON_UNIT
            return factor, "Dobson units"

        # molecules/cm2 -> Dobson units
        if ("molec" in u or "molecule" in u) and ("cm-2" in u or "cm^-2" in u):
            factor = 1.0e4 / MOLEC_PER_DOBSON_UNIT
            return factor, "Dobson units"

        # Unknown: keep original units, no conversion
        return 1.0, units

    def get_current_slice(self):
        """Slice in display units for rendering."""
        return self.data_native * self.display_scale

    def set_current_slice(self, new_slice):
        """Update native data from a rescaled slice."""
        if self.display_scale == 0:
            return
        self.data_native = np.array(new_slice, dtype=float) / self.display_scale
    
    def _get_plot_extent_and_origin(self):
        """Calculate extent (edges) and determine origin based on lat/lon arrays."""
        if self.lats is None or self.lons is None:
             return [0, 1, 0, 1], 'lower'

        # Check for 1D arrays to determine regular grid properties
        if self.lats.ndim == 1 and self.lons.ndim == 1:
            nlat = len(self.lats)
            nlon = len(self.lons)
            
            # Determine lat direction
            if nlat > 1:
                lat_increasing = self.lats[1] > self.lats[0]
                origin = 'lower' if lat_increasing else 'upper'
                
                # Calculate resolution (assuming mostly regular)
                lat_res = abs(self.lats[-1] - self.lats[0]) / (nlat - 1)
                lon_res = abs(self.lons[-1] - self.lons[0]) / (nlon - 1)
                
                # Calculate edges
                extent = [
                    self.lons.min() - lon_res / 2,
                    self.lons.max() + lon_res / 2,
                    self.lats.min() - lat_res / 2,
                    self.lats.max() + lat_res / 2
                ]
                return extent, origin
            else:
                # Fallback for single point or weird data
                return [self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max()], 'lower'
        else:
            # Fallback for 2D/swath data (using min/max centers as approximation)
            # Swath data usually requires pcolormesh for accuracy, but imshow is faster.
            # We'll stick to min/max centers which gives a small offset but usually acceptable for swath if dense.
            # For 'origin', we need to check the general trend of lats along the first dimension.
            
            # Check lat trend along axis 0 (rows)
            origin = 'lower'
            if self.lats.shape[0] > 1:
                # Compare mean of first row vs last row
                 if np.mean(self.lats[0, :]) > np.mean(self.lats[-1, :]):
                     origin = 'upper'

            return [self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max()], origin

    def _colorbar_label(self):
        if self.display_units:
            return f"{self.current_var_name} [{self.display_units}]"
        return self.current_var_name

    def update_plot(self, preserve_view=False):
        """Update the plot with current data"""
        # Get current slice
        current_slice = self.get_current_slice()
        
        # Define extent and origin
        extent, origin = self._get_plot_extent_and_origin()

        # Display the data
        if hasattr(self, 'im') and self.im is not None and self.im.get_array().shape == current_slice.shape:
            self.im.set_data(current_slice)
            self.im.set_extent(extent)
            # origin cannot be updated easily in set_data/set_extent? 
            # Actually imshow origin is fixed at creation. If origin changes (unlikely for same dataset), we might need to recreate.
            # Assuming origin is constant for the dataset files.
            
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.set_label(self._colorbar_label())
        else:
            self.ax.clear()
            self.draw_map_features()
            self.im = self.ax.imshow(current_slice, cmap='Reds', norm=colors.LogNorm(vmin=0.001, vmax=10.0), 
                               origin=origin,
                               transform=ccrs.PlateCarree(),
                               extent=extent)
        
            # Remove the old colorbar if it exists
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except Exception:
                    pass
            
            self.colorbar = plt.colorbar(
                self.im,
                cax=self.cax,
                orientation="vertical",
                extend='both',
                format='%1.2f',
            )
            self.colorbar.set_label(self._colorbar_label())

        # Set axes extent to match data
        if not preserve_view:
            self.ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Update title with dimension information
        title = f"File {self.current_file_idx+1} out of {len(self.nc_files)}"
        self.ax.set_title(title)

        # Show modified indicator
        if self.modified:
            self.fig.suptitle(f"*{os.path.basename(self.nc_files[self.current_file_idx])}*", fontsize=12)
        else:
            self.fig.suptitle(os.path.basename(self.nc_files[self.current_file_idx]), fontsize=12)

        # Update canvas
        self.fig.canvas.draw_idle()

    def get_indices(self, lon, lat):
        """Convert lon, lat to array indices x, y using the linear transform of the plot."""
        if self.lats is None or self.lons is None:
            return None, None
            
        # Get the visual extent and origin used by imshow
        extent, origin = self._get_plot_extent_and_origin()
        lon_min, lon_max, lat_min, lat_max = extent
        
        # Current data shape
        if self.data_native is None:
             return None, None
        
        ny, nx = self.data_native.shape
        
        # Calculate x index (longitude)
        if lon_max == lon_min:
            x_idx = 0
        else:
            x_frac = (lon - lon_min) / (lon_max - lon_min)
            x_idx = int(x_frac * nx)
        
        # Calculate y index (latitude)
        if lat_max == lat_min:
            y_idx = 0
        else:
            y_frac = (lat - lat_min) / (lat_max - lat_min)
            if origin == 'lower':
                y_idx = int(y_frac * ny)
            else: # upper
                y_idx = int((1.0 - y_frac) * ny)
            
        # Clip to bounds
        x_idx = max(0, min(x_idx, nx - 1))
        y_idx = max(0, min(y_idx, ny - 1))
        
        return x_idx, y_idx

    def apply_eraser(self, x_data, y_data):
        """Apply eraser at the given geographic coordinates"""
        if not self.eraser_active:
            return

        if x_data is None or y_data is None:
            return
        
        # Convert geo coords to array indices
        x_idx, y_idx = self.get_indices(x_data, y_data)
        
        if x_idx is None:
             return

        # Get current slice
        current_slice = self.get_current_slice()

        # Create a grid of coordinates (indices)
        ny, nx = current_slice.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        # Calculate distances from click point (in pixels)
        distances = np.sqrt((xx - x_idx)**2 + (yy - y_idx)**2)

        # Set values to NaN where distance is less than radius
        current_slice[distances <= self.eraser_radius] = np.nan

        # Update data
        self.set_current_slice(current_slice)
        self.modified = True

        # Update plot
        self.update_plot(preserve_view=True)

    def update_eraser_radius(self, val):
        """Update eraser radius"""
        self.eraser_radius = int(val)
        # Visual update will happen in on_mouse_move

    def toggle_eraser(self, event):
        """Toggle eraser tool on/off"""
        self.eraser_active = not self.eraser_active

        if self.eraser_active:
            self.eraser_button.label.set_text('Eraser: On')
        else:
            self.eraser_button.label.set_text('Eraser: Off')
            if self.eraser_circle:
                try:
                    self.eraser_circle.remove()
                except NotImplementedError:
                    pass
                self.eraser_circle = None

        self.fig.canvas.draw()

    def on_mouse_move(self, event):
        """Handle mouse movement"""
        # Check if toolbar mode is active (Pan/Zoom)
        if self.fig.canvas.toolbar.mode != '':
            return

        if not event.inaxes or event.inaxes != self.ax:
            if self.eraser_circle:
                try:
                    self.eraser_circle.remove()
                except NotImplementedError:
                    pass
                self.eraser_circle = None
                self.fig.canvas.draw_idle()
            return

        if self.eraser_active:
            # Remove old circle
            if self.eraser_circle:
                try:
                    self.eraser_circle.remove()
                except NotImplementedError:
                    pass
            
            # Calculate radius in degrees for visualization
            # Approximate degrees per pixel
            if self.lats is not None and self.lons is not None:
                lat_min, lat_max = self.lats.min(), self.lats.max()
                lon_min, lon_max = self.lons.min(), self.lons.max()
                
                if self.lats.ndim == 1:
                     ny = len(self.lats)
                     nx = len(self.lons)
                else:
                     ny, nx = self.lats.shape

                deg_per_pix_lat = (lat_max - lat_min) / ny if ny > 0 else 1.0
                deg_per_pix_lon = (lon_max - lon_min) / nx if nx > 0 else 1.0
                avg_deg_per_pix = (deg_per_pix_lat + deg_per_pix_lon) / 2.0
                
                radius_deg = self.eraser_radius * avg_deg_per_pix
            else:
                radius_deg = self.eraser_radius # Fallback

            self.eraser_circle = Circle((event.xdata, event.ydata),
                                       radius_deg,
                                       fill=False,
                                       edgecolor='red',
                                       linestyle='--',
                                       transform=ccrs.PlateCarree()) # Important: circle in data coords
            self.ax.add_patch(self.eraser_circle)
            self.fig.canvas.draw_idle()

            # Apply eraser if mouse button is pressed
            if event.button == 1:  # Left mouse button
                self.apply_eraser(event.xdata, event.ydata)

    def on_mouse_click(self, event):
        """Handle mouse click"""
        # Check if toolbar mode is active (Pan/Zoom)
        if self.fig.canvas.toolbar.mode != '':
            return

        if not event.inaxes or event.inaxes != self.ax:
            return

        if self.eraser_active and event.button == 1:  # Left mouse button
            self.apply_eraser(event.xdata, event.ydata)


    def on_mouse_release(self, event):
        """Handle mouse release"""
        pass

    def save_data(self, event):
        if not self.modified:
            #print("No modifications to save.")
            return

        save_path = self.nc_files[self.current_file_idx]
        dataset = Dataset(save_path, 'r+')
        var = dataset.variables[self.current_var_name]
        if var.ndim == 3:
            var[0, :, :] = self.data_native
        elif var.ndim == 2:
            var[:, :] = self.data_native
        else:
            dataset.close()
            raise ValueError(
                f"Cannot save data: unsupported variable dimensions {var.shape}"
            )
        dataset.close()

        self.modified = False
        self.fig.suptitle(os.path.basename(save_path), fontsize=12)

    def next_file(self, event):
        """Move to the next file, saving changes if needed"""
        # Save if modified
        if self.modified:
            self.save_data(None)

        # Move to next file if available, wrap to start otherwise
        if self.current_file_idx < len(self.nc_files) - 1:
            self.current_file_idx += 1
        else:
            self.current_file_idx = 0
        print (f"Updating the counter of files to {self.current_file_idx}")

        self.load_current_file()
        self.update_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive NetCDF field editor")
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing NetCDF files (prompts if omitted)",
    )
    parser.add_argument(
        "--field",
        help="Name of the NetCDF variable to view/edit (defaults to aerosol_index_354_388)",
    )
    args = parser.parse_args()

    folder_path = args.folder or input("Enter path to folder containing NetCDF files: ")
    viewer = NetCDFViewer(folder_path, field_name=args.field)
