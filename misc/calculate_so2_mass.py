import netCDF4 as nc
import xarray as xr
import numpy as np
import glob
import os

def calculate_total_so2_mass(so2_file_path, wrf_input_file_path):
    # Constants
    MOLAR_MASS_SO2_G_PER_MOL = 64.06  # g/mol
    G_TO_MT = 1e-12                  # 1 Mt = 10^12 g

    print(f"Opening SO2 NetCDF file: {so2_file_path}")
    try:
        so2_data = xr.open_dataset(so2_file_path)
    except FileNotFoundError:
        print(f"Error: SO2 file not found at {so2_file_path}")
        return None
    except Exception as e:
        print(f"Error opening SO2 file: {e}")
        return None

    # Extract SO2 vertical column density
    if 'sulfurdioxide_total_vertical_column_15km' not in so2_data.variables:
        print(f"Error: 'sulfurdioxide_total_vertical_column_15km' not found in {so2_file_path}")
        return None
    
    so2_column = so2_data['sulfurdioxide_total_vertical_column_15km']
    # The variable can have dimensions (Time, south_north, west_east)
    # or (south_north, west_east) if only one time step
    
    # Ensure so2_column is a Dask array if it's not already, to handle potentially large files
    # and perform computations lazily before a final compute().
    if not isinstance(so2_column.data, np.ndarray): # Check if already a numpy array
        so2_column = so2_column.load() # Load into memory if it's not already

    print(f"Opening WRF input file: {wrf_input_file_path}")
    try:
        with nc.Dataset(wrf_input_file_path, 'r') as wrfinput:
            # Extract grid cell dimensions from wrfinput_d01, similar to WRFNetCDFWriter
            MAPFAC_MX = wrfinput.variables['MAPFAC_MX'][0,:]
            MAPFAC_MY = wrfinput.variables['MAPFAC_MY'][0,:]
            DX = wrfinput.getncattr('DX')
            DY = wrfinput.getncattr('DY')
            
            # Calculate cell area in m^2
            # The calculation `(dx/MAPFAC_MX)*(dy/MAPFAC_MY)` from WRFNetCDFWriter
            # implies that MAPFAC_MX and MAPFAC_MY are factors to convert dx/dy to actual distances.
            # Assuming DX and DY are in meters and MAPFAC_MX/MY are unitless scale factors
            area = (DX / MAPFAC_MX) * (DY / MAPFAC_MY) # m^2
            
            # Ensure area matches the spatial dimensions of so2_column
            if area.shape != so2_column.shape[-2:]:
                print(f"Error: Area dimensions {area.shape} do not match SO2 data spatial dimensions {so2_column.shape[-2:]}")
                return None
    except FileNotFoundError:
        print(f"Error: WRF input file not found at {wrf_input_file_path}")
        return None
    except Exception as e:
        print(f"Error opening WRF input file or extracting variables: {e}")
        return None

    # Calculate total moles of SO2
    # so2_column is moles/m^2, area is m^2. Their product is moles.
    total_moles_per_cell = so2_column * area
    
    # Sum over all spatial dimensions (south_north, west_east) and time
    total_moles_so2 = total_moles_per_cell.sum().item() # .item() to get scalar from 0-d array

    # Convert moles to mass in grams
    total_mass_so2_grams = total_moles_so2 * MOLAR_MASS_SO2_G_PER_MOL

    # Convert grams to Mega tons (Mt)
    total_mass_so2_mt = total_mass_so2_grams * G_TO_MT

    return total_mass_so2_mt

if __name__ == "__main__":
    wrf_input_file = 'wrfinput_d01'
    
    so2_files = glob.glob('merged_sulfurdioxide_total_vertical_column_*.nc')
    so2_files.sort() # Process files in a consistent order

    if not so2_files:
        print("No SO2 NetCDF files found matching 'merged_sulfurdioxide_total_vertical_column_*.nc'")
    else:
        for so2_nc_file in so2_files:
            print(f"\n--- Processing file: {os.path.basename(so2_nc_file)} ---")
            mass_mt = calculate_total_so2_mass(so2_nc_file, wrf_input_file)

            if mass_mt is not None:
                print(f"Total mass of SO2: {mass_mt:.6f} Mt")
            print("------------------------------------------")
