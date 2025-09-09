########################################################################
## Mycotoxin maps: per-toxin facet maps + sampling sites              ##
##                                                                    ##
########################################################################
##   • per-toxin concentrations where present (> 0) at sampling sites  ##
##   • locations of all sampling sites                                 ##
## Notes – The underlying dataset is private. This script illustrates  ##
##         the mapping workflow used in the Mycotox-I project.         ##
########################################################################
## Script outline                                                      ##
##   0. Setup: packages, theme                                         ##
##   1. Data: load, identify response cols, back-transform             ##
##   2. Geography: build ROI+NI outline, filter points to island       ##
##   3. Tidy: long format, remove zeros (presence > 0)                 ##
##   4. Scaling: log1p colour scale, custom legend labels              ##
##   5. Plots: (a) sampling-sites map - Figure 1 in paper              ##
##             (b) per-toxin facet map - Figure 3 in paper             ##                         
########################################################################


# ============================== 0. Setup ===============================

library(dplyr)
library(tidyr)
library(ggplot2)
library(sf)
library(viridis)      
library(stringr)
library(rnaturalearth)
library(rnaturalearthdata)
library(tibble)

theme_set(theme_bw(base_size = 12))


# ============================== 1. Data ================================

# Load data
mycdf <- readRDS("mycdf_with_weather.rds")

# Identify response columns (from first toxin to end)
start_resp <- match("Trichothence_producer", names(mycdf))
resp_cols  <- names(mycdf)[start_resp:length(mycdf)]

# Split binary vs continuous
bin_vars  <- resp_cols[str_detect(resp_cols, "_bin$")]
cont_vars <- setdiff(resp_cols, bin_vars)

# Back-transform continuous responses if stored as log1p
mycdf <- mycdf |>
  mutate(across(all_of(cont_vars), ~ expm1(.x)))

# Display names for facets
clean_names <- c(
  "Trichothence_producer" = "Trichothecene Prod.",
  "F_langsethiae" = "F. langsethiae",
  "F_poae" = "F. poae",
  "DON" = "DON",
  "D3G" = "D3G",
  "Nivalenol" = "Nivalenol",
  "3-AC-DON" = "3-AC-DON",
  "15-AC-DON" = "15-AC-DON",
  "T-2_toxin" = "T-2 Toxin",
  "HT-2_toxin" = "HT-2 Toxin",
  "T2G" = "T2G",
  "Neos" = "Neosolaniol",
  "ENN_A1" = "ENN A1",
  "ENN_A"  = "ENN A",
  "ENN_B"  = "ENN B",
  "ENN_B1" = "ENN B1",
  "BEAU" = "Beauvericin",
  "ZEN"  = "Zearalenone",
  "Apicidin" = "Apicidin",
  "STER" = "Sterigmatocystin",
  "DAS"  = "Diacetoxyscirpenol",
  "Quest" = "Questin",
  "AOH"  = "Alternariol",
  "AME"  = "AME",
  "MON"  = "Moniliformin",
  "Ergocristine" = "Ergocristine",
  "EGT" = "Ergot alkaloids"
)

# Columns for coords and site ID
site_id_col <- "Location_details"
lon_col     <- "Longitude"
lat_col     <- "Latitude"

# Presence threshold (keep values strictly > 0)
presence_threshold <- 0


# ============================ 2. Geography =============================

# ROI and NI outlines
irl_country <- ne_countries(country = "Ireland", scale = 10, returnclass = "sf") |> st_make_valid()
uk_states   <- ne_states(country = "United Kingdom", returnclass = "sf") |> st_make_valid()
ni_region   <- uk_states |>
  dplyr::filter(if_any(where(is.character),
                       ~ grepl("\\bNorthern Ireland\\b", .x, ignore.case = TRUE)))

ireland_outline <- bind_rows(
  irl_country |> dplyr::select(geometry),
  ni_region   |> dplyr::select(geometry)
) |>
  st_union() |>
  st_make_valid() |>
  st_as_sf(crs = 4326)

# Sampling points as sf, keep points intersecting the island; small buffer retains near-coast points
pts <- mycdf |>
  dplyr::filter(!is.na(.data[[lon_col]]), !is.na(.data[[lat_col]])) |>
  st_as_sf(coords = c(lon_col, lat_col), crs = 4326, remove = FALSE)

pts_in <- st_filter(pts, st_buffer(ireland_outline, dist = 0.01), .predicate = st_intersects)


# ============================== 3. Tidy ================================

# Toxins to plot (continuous names)
mycotoxin_cols <- c(
  "Trichothence_producer","F_langsethiae","F_poae","DON","D3G",
  "Nivalenol","3-AC-DON","15-AC-DON","T-2_toxin","HT-2_toxin",
  "T2G","Neos","ENN_A1","ENN_A","ENN_B","ENN_B1","BEAU",
  "ZEN","Apicidin","STER","DAS","Quest","AOH","AME","MON",
  "Ergocristine","EGT"
)

# Long format with clean labels
map_df <- tibble(toxin = names(clean_names),
                 toxin_clean = unname(clean_names))

long_all <- mycdf |>
  dplyr::select(all_of(c(site_id_col, lon_col, lat_col)), all_of(mycotoxin_cols)) |>
  tidyr::pivot_longer(cols = all_of(mycotoxin_cols),
                      names_to = "toxin", values_to = "value") |>
  dplyr::mutate(value = as.numeric(value)) |>
  dplyr::left_join(map_df, by = "toxin")

# Keep present only 
long_present <- long_all |>
  dplyr::filter(!is.na(value), value > presence_threshold)

# Drop toxins with zero detections (avoid empty facets)
detected_toxins <- long_present |>
  dplyr::distinct(toxin, toxin_clean)

long_present <- long_present |>
  dplyr::semi_join(detected_toxins, by = c("toxin","toxin_clean"))


# ============================= 4. Scaling ==============================

# Map colour to log1p(value) but label legend in original units
long_present <- long_present |>
  dplyr::mutate(val_col = log1p(value))

colour_name <- "Concentration (µg/kg)"

# Legend labels based on observed range in original units
max_raw_val <- max(long_present$value, na.rm = TRUE)
min_raw_val <- min(long_present$value, na.rm = TRUE)

# Sequence 10^k within range
legend_labels <- 10^seq(floor(log10(min_raw_val)), ceiling(log10(max_raw_val)), by = 1)
legend_labels <- round(legend_labels, 0)
legend_labels <- legend_labels[-c(1:3)]
legend_labels <- legend_labels[-length(legend_labels)]
legend_labels <- unique(c(0, legend_labels)) |> sort()
legend_labels <- legend_labels[legend_labels != 1]
legend_labels <- legend_labels[legend_labels <= max_raw_val]

formatted_labels <- sprintf("%.0f", legend_labels)
legend_breaks   <- log1p(legend_labels)

# two-colour palette 
temp_colors <- c("navy", "yellow")
pal  <- grDevices::colorRampPalette(temp_colors)
cols <- pal(100)


# ============================== 5. Plots ===============================

# (a) Sampling sites map - Figure 1 in paper
map_sites <- ggplot() +
  geom_sf(data = ireland_outline, fill = NA, colour = "grey20", linewidth = 0.6) +
  geom_point(
    data = pts_in,
    aes(x = .data[[lon_col]], y = .data[[lat_col]]),
    size = 2,
    colour = "navyblue",
    alpha = 0.8
  ) +
  coord_sf(xlim = c(-11.2, -5.2), ylim = c(51.2, 55.6), expand = FALSE) +
  labs(x = "Longitude", y = "Latitude") +
  theme(panel.grid.major = element_line(colour = "grey85", linewidth = 0.2))

# Print sampling sites map
map_sites

# (b) Per-toxin facet map - Figure 3 in paper
p_facets_present <- ggplot() +
  geom_sf(data = ireland_outline, fill = NA, colour = "grey20", linewidth = 0.4) +
  geom_point(
    data = long_present,
    aes(x = .data[[lon_col]], y = .data[[lat_col]], colour = val_col),
    alpha = 0.5, size = 1.7
  ) +
  scale_colour_gradientn(
    colours = cols,
    name    = colour_name,
    breaks  = legend_breaks,
    labels  = formatted_labels,
    limits  = c(0, max(long_present$val_col)),
    oob     = scales::squish,
    guide   = guide_colorbar(frame.colour = "black", ticks.colour = "black")
  ) +
  coord_sf(xlim = c(-11.2, -5.2), ylim = c(51.2, 55.6), expand = FALSE) +
  facet_wrap(~ toxin_clean, ncol = 5, scales = "fixed", drop = TRUE) +
  labs(x = "Longitude", y = "Latitude") +
  scale_x_continuous(n.breaks = 1) +
  theme(
    panel.grid.major = element_line(colour = "grey90", linewidth = 0.2),
    strip.text       = element_text(size = 10),
    strip.background = element_rect(fill = "grey95")
  )

# change facet ncol and axis breaks
p_facets_present +
  facet_wrap(~ toxin_clean, ncol = 9) +
  scale_x_continuous(breaks = c(-11, -9, -7, -5)) +
  scale_y_continuous(breaks = c(52, 54, 56))
