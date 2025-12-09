import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import ndimage

# The file paths are passed as command-line arguments
file_paths = [
    "../4km/ash5_run/mass_emission_time_height.txt",
    "../4km/ash6_run/mass_emission_time_height.txt",
    "../4km/ash7_run/mass_emission_time_height.txt",
    "../4km/ash8_run/mass_emission_time_height.txt",
    "../4km/ash9_run/mass_emission_time_height.txt",
    "../4km/ash10_run/mass_emission_time_height.txt",
]

# Read and parse each file into a numpy array, skipping the header rows
arrays = []
for file_path in file_paths:
    arr = np.loadtxt(file_path, skiprows=2)
    arrays.append(arr)
    print(f"Sum of array from {file_path}: {np.sum(arr)}\n")

# Sum the arrays
total_array = np.sum(arrays, axis=0)


# Get the header from the first file
with open(file_paths[0]) as f:
    header = [next(f) for _ in range(2)]

# Write the header and the summed array to the output file
output_filename = "combined_ash.txt"
with open(output_filename, "w") as f:
    for line in header:
        f.write(line)

with open(output_filename, "a") as f:
    np.savetxt(f, total_array, fmt="%d")

print(f"Successfully saved the summed array with headers to {output_filename}")

# Read the data from the file
with open("combined_ash.txt") as f:
    time_line = f.readline()
    height_line = f.readline()
    data = np.loadtxt(f)
data = np.flipud(data) # Flip the 2D array vertically

# Parse the time and height values
times_str = time_line.split()[1:]
heights_str = height_line.split()[1:]

times = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in times_str]
heights = [float(h) for h in heights_str]

# Calculate total number of parcels
total_parcels = np.sum(data)

# Create a custom colormap starting with white for the lowest values
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Get colors from viridis_r
viridis_r_cmap = plt.colormaps['viridis_r']
new_colors = viridis_r_cmap(np.linspace(0, 1, 256))
# Make the first color white (for the lowest data values)
new_colors[0, :] = np.array([1, 1, 1, 1]) # RGBA for white
cmap = LinearSegmentedColormap.from_list("white_viridis_r", new_colors, N=256)

# Define the levels for the colorbar
# Ensure data.min() is included in the lowest bin
levels = MaxNLocator(nbins=15).tick_values(data.min(), data.max())
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
mesh = ax.pcolormesh(times, heights, data, shading='auto', cmap=cmap, norm=norm, edgecolors=(0, 0, 0, 0.3), linewidths=0.5)

# Find the two largest connected components and create a contour around them
# Create a binary version of the data
binary_data = data > 0
# Label connected components
labeled_array, num_features = ndimage.label(binary_data)
# Calculate the area of each component
areas = ndimage.sum_labels(binary_data, labeled_array, range(1, num_features + 1))
# Find the labels of the two largest components
# Add a check to handle cases with fewer than two components
if len(areas) > 1:
    two_largest_labels = np.argsort(areas)[-2:] + 1 # +1 to match label numbers
    # Create a mask for the two largest components
    mask = np.isin(labeled_array, two_largest_labels)
    # Draw a contour around the mask
    ax.contour(times, heights, mask, levels=[0.5], colors='black', linewidths=1)
elif len(areas) > 0:
    # Handle case with only one component
    largest_label = np.argsort(areas)[-1] + 1
    mask = labeled_array == largest_label
    ax.contour(times, heights, mask, levels=[0.5], colors='black', linewidths=1)

# Format the axes
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Height (km)", fontsize=12) # Change label to km
cbar = fig.colorbar(mesh, ax=ax, ticks=levels) # Pass levels to colorbar for discrete ticks
cbar.set_label('Number of Parcels')

# Set y-ticks and labels in km
ax.set_yticks(heights)
ax.set_yticklabels([f"{h/1000:.1f}" for h in heights]) # Convert m to km and format

# Set the title with the total number of parcels
ax.set_title(f"Total Parcels: {int(total_parcels)}")

# Format the x-axis to display dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation='vertical')

# Generate x-axis tick locations for 1.5 hour intervals starting from 08:30
start_time = datetime(times[0].year, times[0].month, times[0].day, 8, 30)
tick_locations = []
current_time = start_time
while current_time <= times[-1]:
    tick_locations.append(current_time)
    current_time += timedelta(minutes=90) # Add 1 hour 30 minutes

ax.set_xticks(tick_locations) # Set custom tick locations
ax.set_xlim(times[0], times[-1]) # Set x-axis limits to match data range

# Save the plot
output_plot_filename = "combined_ash.png"
plt.savefig(output_plot_filename, bbox_inches='tight')

print(f"Successfully created plot and saved to {output_plot_filename}")

print(f"Total sum of all parcels: {int(total_parcels)}")
