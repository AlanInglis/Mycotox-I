########################################################################
## ERA5 Ireland: download, process, and visualise climate rasters      ##
##                                                                    ##
########################################################################
## Aim –                                                               ##
##   • fetch ERA5 total precipitation and 2 m temperature by year      ##
##   • aggregate to annual totals/means and resample to fine grid      ##
##   • mask to Ireland (ROI + NI) and build per-year maps              ##
##   • compose a 2×2 summary figure (precip 2022/2023, temp 2022/2023) ##
## Notes – Requires CDS credentials (UID/API key) via ecmwfr. Paths    ##
##         are local; this script demonstrates the mapping workflow.    ##
########################################################################
## Script outline                                                      ##
##   1. Packages                                                       ##
##   2. ERA5 auth + download helper                                    ##
##   3. Spatial helpers: Ireland outline, NetCDF → annual raster       ##
##   4. Plot helper                                                    ##
##   5. Main loop: download (optional), process, plot per year/var     ##
##   6. Combine 4 panels with patchwork                                ##
########################################################################


# ---- 1. Load Packages ----

library(ecmwfr)
library(keyring)
library(terra)
library(sf)
library(dplyr)
library(ggplot2)
library(viridis)
library(rnaturalearth)
library(rnaturalearthdata)
library(patchwork)


# ---- 2. ERA5 Credentials and Data Download (Example) ----

# This section provides an example of how to authenticate with and download data
# from the Copernicus Climate Data Store (CDS).
#
# IMPORTANT: Replace "YOUR_UID" and "YOUR_API_KEY" with your actual
# CDS UID and API key. You can find these on your CDS profile page.

# wf_set_key(user = "YOUR_UID", key = "YOUR_API_KEY")

# Define a function to download ERA5 data for a given year and variable.
download_era5_data <- function(year, variable, target_file, path = getwd()) {
  
  # Return early if the file already exists to avoid re-downloading.
  if (file.exists(file.path(path, target_file))) {
    message("File '", target_file, "' already exists. Skipping download.")
    return(invisible(NULL))
  }
  
  message("Submitting ERA5 request for ", variable, " in ", year, ".")
  
  # Bounding box for Ireland [North, West, South, East]
  ireland_bbox <- c(55.5, -11, 51.3, -5.4)
  
  # Construct the request list for the CDS API.
  request <- list(
    dataset_short_name = "reanalysis-era5-single-levels",
    product_type       = "reanalysis",
    variable           = variable,
    year               = as.character(year),
    month              = sprintf("%02d", 1:12),
    day                = sprintf("%02d", 1:31),
    time               = sprintf("%02d:00", 0:23),
    area               = ireland_bbox,
    format             = "netcdf",
    target             = target_file
  )
  
  # Submit the data request.
  # This requires your UID and API key to be set via wf_set_key().
  # The actual UID is passed here from the placeholder at the top.
  wf_request(
    user     = "YOUR_UID", # Replace with your UID
    request  = request,
    transfer = TRUE,
    path     = path
  )
  
  message("Download complete for '", target_file, "'.")
}


# ---- 3. Spatial Data Processing Functions ----


# Function to create a high-resolution outline of Ireland (ROI + NI).
create_ireland_outline <- function() {
  message("Creating Ireland map outline...")
  irl_country <- ne_countries(country = "Ireland", scale = "large", returnclass = "sf")
  uk_states   <- ne_states(country = "United Kingdom", returnclass = "sf")
  
  ni_region   <- uk_states |>
    filter(grepl("Northern Ireland", name, ignore.case = TRUE))
  
  # Combine geometries and ensure validity.
  st_union(irl_country$geometry, ni_region$geometry) |>
    st_as_sf() |>
    st_make_valid()
}

# Function to process raw ERA5 data: load, aggregate, resample, and mask.
process_era5_data <- function(file_path, variable, ireland_outline_sf) {
  message("Processing ", variable, " data from '", basename(file_path), "'...")
  
  # Load the downloaded NetCDF file as a SpatRaster.
  raw_raster <- rast(file_path)
  
  # Process based on the variable type.
  if (variable == "total_precipitation") {
    # Sum hourly precipitation (in meters) to get annual total.
    processed_raster <- sum(raw_raster, na.rm = TRUE) * 1000 # Convert m to mm
  } else if (variable == "2m_temperature") {
    # Calculate the mean of hourly temperatures (in Kelvin).
    processed_raster <- mean(raw_raster, na.rm = TRUE) - 273.15 # Convert K to °C
  } else {
    stop("Unsupported variable type: ", variable)
  }
  
  # Reproject the Ireland outline to match the raster's CRS.
  ireland_outline_reproj <- st_transform(ireland_outline_sf, crs = st_crs(processed_raster))
  
  # Resample raster to a finer resolution (0.01 degrees) for a smoother plot.
  ireland_outline_vect <- vect(ireland_outline_reproj)
  template <- rast(ext(ireland_outline_vect), resolution = 0.01, crs = crs(processed_raster))
  annual_hi_res <- resample(processed_raster, template, method = "bilinear")
  
  # Mask the high-resolution raster to the exact shape of Ireland.
  mask_raster <- rasterize(ireland_outline_vect, template, cover = TRUE)
  annual_clipped <- terra::mask(annual_hi_res, mask_raster)
  
  # Convert to a data frame for ggplot2.
  df <- as.data.frame(annual_clipped, xy = TRUE, na.rm = TRUE)
  names(df) <- c("lon", "lat", "value")
  
  return(df)
}


# ---- 4. Plotting Function ----


# Function to generate a standardized map plot.
create_era5_plot <- function(df, ireland_outline_sf, legend_title, color_palette) {
  message("Generating plot for '", legend_title, "'...")
  
  ireland_bbox <- c(55.5, -11, 51.3, -5.4) # N, W, S, E
  
  ggplot() +
    geom_raster(data = df, aes(x = lon, y = lat, fill = value)) +
    geom_sf(data = ireland_outline_sf, fill = NA, colour = "black", linewidth = 0.4) +
    scale_fill_gradientn(
      colours = color_palette,
      name    = legend_title,
      guide   = guide_colorbar(frame.colour = "black", ticks.colour = "black")
    ) +
    coord_sf(xlim = ireland_bbox[c(2, 4)], ylim = ireland_bbox[c(3, 1)], expand = FALSE) +
    labs(x = "Longitude", y = "Latitude") +
    theme_bw()
}


# ---- 5. Main Analysis Workflow ----


# Define years and variables to analyze.
years <- c(2022, 2023)
variables <- c("total_precipitation", "2m_temperature")

# Define color palettes for each variable.
precip_colors <- colorRampPalette(c("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"))(100)
temp_colors   <- colorRampPalette(c("navy", "yellow"))(100)

# Create the Ireland outline once to be reused in all plots.
ireland_outline <- create_ireland_outline()

# A list to store the generated plots.
plot_list <- list()

# Loop through each year and variable to download, process, and plot the data.
for (year in years) {
  for (variable in variables) {
    
    # --- Data Download ---
    target_file <- paste0("era5_ireland_", year, "_", variable, ".nc")
    # UNCOMMENT THE LINE BELOW TO ENABLE DOWNLOADING
    # download_era5_data(year, variable, target_file)
    
    # Check if the file exists before proceeding.
    if (!file.exists(target_file)) {
      warning("File '", target_file, "' not found. Skipping processing and plotting.", call. = FALSE)
      next
    }
    
    # --- Data Processing ---
    plot_df <- process_era5_data(target_file, variable, ireland_outline)
    
    # --- Plot Generation ---
    if (variable == "total_precipitation") {
      legend_title <- paste0("Annual\nRainfall (mm)\n", year)
      colors <- precip_colors
    } else {
      legend_title <- paste0("Annual Mean\nTemp (°C)\n", year)
      colors <- temp_colors
    }
    
    # Create the plot and add it to our list.
    current_plot <- create_era5_plot(plot_df, ireland_outline, legend_title, colors)
    plot_key <- paste(variable, year, sep = "_")
    plot_list[[plot_key]] <- current_plot
  }
}


# ---- 6. Combine and Display Plots ----
# Figure 3 in paper
# Arrange the four generated plots into a 2x2 grid using the patchwork library.

if (length(plot_list) == 4) {
  message("Arranging final combined plot...")
  
  # Arrange plots in a specific order: Precip 2022, Precip 2023, Temp 2022, Temp 2023
  combined_plot <- (plot_list[["total_precipitation_2022"]] | plot_list[["total_precipitation_2023"]]) /
    (plot_list[["2m_temperature_2022"]]    | plot_list[["2m_temperature_2023"]])
  
  # Display the final composite image.
  print(combined_plot)
  
  message("Analysis complete.")
} else {
  warning("Could not generate all 4 plots. Final combined plot will not be created.", call. = FALSE)
}