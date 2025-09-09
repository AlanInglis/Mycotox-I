########################################################################
## Exceedance maps: EU thresholds (oats) + infant-food thresholds     ##
##                                                                    ##
########################################################################
##   • compute threshold exceedances per toxin                         ##
##   • map only sites exceeding limits (faceted by toxin)              ##
##   • show standard EU limits and stricter infant-food limits         ##
## Notes – Data are private. This script illustrates exceedance maps   ##
##         for the Mycotox-I project; behaviour preserved.             ##
########################################################################
## Script outline                                                      ##
##   0. Setup: packages, data                                          ##
##   1. Back-transform continuous toxins                               ##
##   2. EU oats limits: exceedance summary + faceted map               ##
##   3. Infant-food limits: exceedance summary + faceted map           ##
##   4. Compose final figure (standard vs infant)                      ##
########################################################################


# ============================== 0. Setup ===============================

library(tidyverse)
library(sf)
library(rnaturalearth)
library(patchwork)

# Load data
mycdf <- readRDS("/Users/alaninglis/Desktop/MycotoxI_App/mycdf_with_weather.rds")

# Ireland outline (ROI + NI)
irl_country <- ne_countries(country = "Ireland", scale = 10, returnclass = "sf") |> st_make_valid()
uk_states   <- ne_states(country = "United Kingdom", returnclass = "sf") |> st_make_valid()
ni_region   <- uk_states |>
  dplyr::filter(if_any(where(is.character), ~ grepl("\\bNorthern Ireland\\b", .x, ignore.case = TRUE)))
ireland_outline <- bind_rows(
  irl_country |> dplyr::select(geometry),
  ni_region   |> dplyr::select(geometry)
) |>
  st_union() |> st_make_valid() |> st_as_sf(crs = 4326)


# ================= 1. Back-transform continuous toxins =================

cont_vars <- c("DON","D3G","Nivalenol","3-AC-DON","15-AC-DON","T-2_toxin",
               "HT-2_toxin","T2G","Neos","ENN_A1","ENN_A","ENN_B","ENN_B1",
               "BEAU","ZEN","Apicidin","STER","DAS","Quest","AOH","AME",
               "MON","Ergocristine","EGT")

mycdf <- mycdf |>
  mutate(across(all_of(cont_vars), ~ expm1(.x)))


# ======= Shared helpers  =======

# Long-format helper
toxin_long <- function(df, toxin_names, clean_names, site_id = "Location_details",
                       lon = "Longitude", lat = "Latitude") {
  map_df <- tibble(toxin = names(clean_names), toxin_clean = unname(clean_names))
  df |>
    dplyr::select(all_of(c(site_id, lon, lat)), all_of(toxin_names)) |>
    tidyr::pivot_longer(cols = all_of(toxin_names),
                        names_to = "toxin", values_to = "value") |>
    dplyr::left_join(map_df, by = "toxin")
}

# Plotting helper (plots only exceeding points)
plot_exceedance_facets <- function(df_long, thresholds_vec, clean_names_vec,
                                   lon = "Longitude", lat = "Latitude") {
  thr_df <- enframe(thresholds_vec, name = "toxin", value = "threshold") |>
    left_join(tibble(toxin = names(clean_names_vec),
                     toxin_clean = unname(clean_names_vec)), by = "toxin")
  
  exceeding_points <- df_long |>
    left_join(thr_df, by = c("toxin", "toxin_clean")) |>
    filter(!is.na(threshold) & value >= threshold)
  
  if (nrow(exceeding_points) == 0) {
    return(
      ggplot() +
        geom_sf(data = ireland_outline, fill = "grey90", colour = "grey20") +
        labs(title = "No sites exceeded the specified thresholds.") +
        theme_bw()
    )
  }
  
  ggplot() +
    geom_sf(data = ireland_outline, fill = NA, colour = "grey20", linewidth = 0.3) +
    geom_point(
      data = exceeding_points,
      aes(x = .data[[lon]], y = .data[[lat]]),
      shape = 18, color = "firebrick", size = 4, alpha = 0.9
    ) +
    coord_sf(xlim = c(-11.2, -5.2), ylim = c(51.2, 55.6), expand = FALSE) +
    facet_wrap(~ toxin_clean) +
    labs(x = "Longitude", y = "Latitude") +
    theme_bw()
}


# ============ 2. EU oats limits: exceedance + faceted map ============

# Combined T-2 + HT-2
mycdf <- mycdf |> mutate(T2_HT2_sum = `T-2_toxin` + `HT-2_toxin`)

mycotoxin_cols_regulated <- c("DON", "ZEN", "T2_HT2_sum", "EGT")
clean_names_regulated <- c(
  "DON" = "DON",
  "ZEN" = "Zearalenone",
  "T2_HT2_sum" = "T-2 + HT-2 Toxins",
  "EGT" = "Ergot Alkaloids"
)

# EU regulatory thresholds for oats (µg/kg)
thresholds <- c("DON" = 1750, "ZEN" = 100, "T2_HT2_sum" = 1000, "EGT" = 150)

# Long
long_df <- toxin_long(mycdf, mycotoxin_cols_regulated, clean_names_regulated)

# Summary 
exceedance_data <- long_df |>
  left_join(enframe(thresholds, name = "toxin", value = "threshold"), by = "toxin") |>
  mutate(exceeds = value >= threshold)

counts_per_toxin <- exceedance_data |>
  filter(exceeds) |>
  count(toxin = factor(clean_names_regulated[toxin], levels = unname(clean_names_regulated)),
        name = "Number of Exceeding Sites")

total_exceeding_locations <- exceedance_data |>
  filter(exceeds) |>
  distinct(Location_details) |>
  nrow()

cat("--- Mycotoxin Exceedance Summary ---\n")
cat("Total sites sampled:", n_distinct(mycdf$Location_details), "\n\n")
if (nrow(counts_per_toxin) > 0) print(as.data.frame(counts_per_toxin)) else
  cat("No sites exceeded the threshold for any mycotoxin.\n")
cat("\nTotal unique locations with at least one exceedance:", total_exceeding_locations, "\n")
cat("--------------------------------------\n\n")

# Map
p_exc <- plot_exceedance_facets(long_df, thresholds, clean_names_regulated)


# ===== 3. Infant-food limits: exceedance + faceted map (strict) ======

mycotoxin_cols_infant <- c("DON","ZEN","HT-2_toxin")
clean_names_infant <- c(
  "DON" = "DON",
  "ZEN" = "Zearalenone",
  "HT-2_toxin" = "HT-2 Toxin"
)
# Infant-food regulatory thresholds (µg/kg)
thresholds_infant <- c("DON" = 200, "ZEN" = 20, "HT-2_toxin" = 10)

# Long
long_df_infant <- toxin_long(mycdf, mycotoxin_cols_infant, clean_names_infant)

# Summary (printed)
exceedance_data_infant <- long_df_infant |>
  left_join(enframe(thresholds_infant, name = "toxin", value = "threshold"), by = "toxin") |>
  mutate(exceeds = value >= threshold)

counts_per_toxin_infant <- exceedance_data_infant |>
  filter(exceeds) |>
  count(toxin = factor(clean_names_infant[toxin], levels = unname(clean_names_infant)),
        name = "Number of Exceeding Sites")

total_exceeding_locations_infant <- exceedance_data_infant |>
  filter(exceeds) |>
  distinct(Location_details) |>
  nrow()

cat("--- Mycotoxin Exceedance Summary (vs. INFANT FOOD Limits) ---\n")
cat("Total sites sampled:", n_distinct(mycdf$Location_details), "\n\n")
if (nrow(counts_per_toxin_infant) > 0) print(as.data.frame(counts_per_toxin_infant)) else
  cat("No sites exceeded the infant food threshold for any mycotoxin.\n")
cat("\nTotal unique locations with at least one exceedance:", total_exceeding_locations_infant, "\n")
cat("----------------------------------------------------------\n\n")

# Map
p_exc_infant <- plot_exceedance_facets(long_df_infant, thresholds_infant, clean_names_infant)


# =========================== 4. Compose plot ===========================

# Figure 4 in paper
p_exc / p_exc_infant
