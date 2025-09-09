##############################################################################
# Mycotox-I: Agronomic & Weather Data Processing Pipeline                    #
##############################################################################
# This script provides an end-to-end workflow for cleaning, processing, and  #
# feature engineering of the Mycotox-I project dataset. It integrates raw    #
# agronomic and mycotoxin data with historical weather data to produce a     #
# final, model-ready dataset.                                                #
#                                                                            #
# Key Processing Stages:                                                     #
#   1. Raw Data Cleaning: Tidies formats, handles missing values, and        #
#      standardises units.                                                   #
#   2. Imputation: Uses MICE with random forests to fill missing predictor   #
#      values.                                                               #
#   3. Weather Integration: Matches samples to the nearest weather station   #
#      and engineers 90-day lagged weather features (rain, temp, RH).        #
#   4. Feature Engineering: Transforms dates to cyclical features, one-hot   #
#      encodes categorical variables, and adds binary toxin outcomes.        #
#   5. Normalisation: Scales numeric predictors to prepare the data for      #
#      machine learning models                                               #
##############################################################################


# ---- 0. Load Packages ----
library(tidyverse)
library(lubridate)
library(mice)
library(geosphere)
library(recipes)


# ---- 1. Raw Data Cleaning & Preparation ----

# ── 1.1. Read and Tidy Initial Data ───────────────────────────────────────────
# Read the raw data file.
mycdf <- read_csv("Mycotoxi_Data.csv", show_col_types = FALSE)

# Clean column names by replacing dots, spaces, and repeated underscores with a single underscore.
mycdf <- mycdf %>%
  rename_with(~ .x %>%
                str_replace_all("[\\._ ]+", "_") %>%
                str_replace_all("^_|_$", ""))

# ── 1.2. Handle Missing Value Codes ───────────────────────────────────────────
# Replace a specific numeric code (-777) with 0, as per data dictionary rules.
mycdf[mycdf == -777] <- 0

# Define common missing value codes used in the dataset.
missing_codes <- c("-999", "-888", "-666", "-555", "-444")

# Replace all missing codes with NA, respecting column data types.
mycdf <- mycdf %>%
  mutate(across(everything(), ~ replace(.x, .x %in% missing_codes, NA)))

# ── 1.3. Remove Redundant Rows and Columns ───────────────────────────────────
# Remove unneeded summary columns, columns that are entirely NA, and numeric columns that are entirely zero.
mycdf <- mycdf %>%
  select(
    -matches("_(Mean|SE|SD)$"),
    where(~ !all(is.na(.))),
    where(~ !(is.numeric(.) && all(. == 0, na.rm = TRUE)))
  )

# Identify response variables (from 'Trichothence_producer' to the end).
response_vars <- names(mycdf)[which(names(mycdf) == "Trichothence_producer"):ncol(mycdf)]

# Remove rows that have no toxin data at all.
mycdf <- mycdf %>%
  filter(if_any(all_of(response_vars), ~ !is.na(.)))

# ── 1.4. Standardise Data Types and Text ─────────────────────────────────────

# Convert all integer columns to numeric to ensure consistency.
mycdf <- mycdf %>%
  mutate(across(where(is.integer), as.numeric))

# Define categorical columns for text cleaning.
cat_cols <- c(
  "Sample_code", "Institute", "Nature", "Origin", "County", "Location_details",
  "Crop", "Sowing_Ideotype", "Variety", "Rotation", "Establishment_system",
  "Cropping_system", "Soil_type", "Previous_year_1", "Previous_year_2",
  "Previous_year_3", "Previous_year_4", "Previous_year_5",
  "Fertiliser_product_1", "Fertiliser_product_2", "Micronutrients_product",
  "Herbicide_product_1", "Herbicide_product_2", "Insecticide_product",
  "Fungicide_product_1", "Fungicide_product_2", "Fungicide_product_3", "Fungicide_product_4",
  "Growth_regulator_product_1", "Growth_regulator_product_2", "Molluscicide_product"
)

# Clean and factorize categorical columns: trim whitespace, convert to Title Case.
mycdf <- mycdf %>%
  mutate(across(
    .cols = all_of(intersect(cat_cols, names(.))),
    .fns = ~ .x %>% as.character() %>% str_trim() %>% str_to_lower() %>% str_to_title() %>% factor()
  ))

# ── 1.5. Parse and Standardise Date Columns ──────────────────────────────────

# Define columns expected to contain dates.
date_cols <- c(
  "Sowing_date", "Fertiliser_application_time_1", "Fertiliser_application_time_2",
  "Micronutrients_application_date", "Herbicide_application_time_1", "Herbicide_application_time_2",
  "Insecticide_application_time", "Fungicide_application_time_1", "Fungicide_application_time_2",
  "Fungicide_application_time_3", "Fungicide_application_time_4",
  "Growth_regulator_application_time_1", "Growth_regulator_application_time_2", "Harvest"
)

# Function to parse dates from multiple potential formats (e.g., ymd, dmy, with times).
parse_mixed_date <- function(x) {
  x <- as.character(x) %>%
    str_trim() %>%
    na_if("") %>%
    # Fix invalid dates like "2022-05-00" by changing day to "01".
    str_replace("(\\d{4}-\\d{2})-00(?!\\d)", "\\1-01")
  
  as_date(parse_date_time(x, orders = c("ymd", "dmy", "ymd_HMS", "dmy_HMS", "ymd_HM", "dmy_HM"), quiet = TRUE))
}

mycdf <- mycdf %>%
  mutate(across(all_of(intersect(date_cols, names(.))), parse_mixed_date))

# ── 1.6. Clean and Correct Specific Columns ──────────────────────────────────
# Convert Soil_pH ranges (e.g., "6.5-7.0") to their midpoint.
mycdf <- mycdf %>%
  mutate(Soil_pH = {
    ph_str <- as.character(Soil_pH)
    lo <- as.numeric(str_extract(ph_str, "^[0-9.]+"))
    hi <- as.numeric(str_extract(ph_str, "(?<=-)[0-9.]+"))
    if_else(!is.na(hi), (lo + hi) / 2, lo)
  })

# Remove columns with inconsistent units that cannot be programmatically resolved.
# For example, herbicide and micronutrient doses have mixed or missing units.
mycdf <- mycdf %>%
  select(-matches("herbicide"), -matches("micronutrients"))

# Standardise columns where units are present in cells (e.g., "1.5 l/ha").
# Convert values to numeric and remove the unit text.
unit_pattern_l_ha <- regex("\\s*l/ha", ignore_case = TRUE)
cols_with_l_ha <- mycdf %>%
  summarise(across(everything(), ~ any(str_detect(as.character(.x), unit_pattern_l_ha)))) %>%
  pivot_longer(everything()) %>%
  filter(value) %>%
  pull(name)

mycdf <- mycdf %>%
  mutate(across(all_of(cols_with_l_ha), ~ as.numeric(str_remove_all(as.character(.x), unit_pattern_l_ha))))

# Find columns with 'ml/ha', remove the unit, and convert the value to 'l/ha'.
unit_pattern_ml_ha <- regex("\\s*ml/ha", ignore_case = TRUE)
cols_with_ml_ha <- mycdf %>%
  summarise(across(everything(), ~ any(str_detect(as.character(.x), unit_pattern_ml_ha)))) %>%
  pivot_longer(everything()) %>%
  filter(value) %>%
  pull(name)

mycdf <- mycdf %>%
  mutate(across(all_of(cols_with_ml_ha), ~ as.numeric(str_remove_all(as.character(.x), unit_pattern_ml_ha)) / 1000))

# Remove rows for "Medax Max" growth regulator, which uses kg/ha instead of the standard l/ha.
mycdf <- mycdf %>%
  filter(
    is.na(Growth_regulator_product_1) | str_to_lower(Growth_regulator_product_1) != "medax max",
    is.na(Growth_regulator_product_2) | str_to_lower(Growth_regulator_product_2) != "medax max"
  )

# Convert soil element index columns to ordered factors. 'Magnesium' is excluded due to out-of-range values.
cat_elements <- c("Phosphorus", "Potassium", "Manganese", "Copper", "Zinc")
mycdf <- mycdf %>%
  select(-any_of("Magnesium")) %>%
  mutate(across(all_of(intersect(cat_elements, names(.))), ~ factor(.x, levels = 1:4, ordered = TRUE)))

# Fix 'Fertiliser_application_rate_1', which is character due to non-numeric entries.
mycdf <- mycdf %>%
  mutate(Fertiliser_application_rate_1 = as.numeric(Fertiliser_application_rate_1))

# ── 1.7. Log-transform Response Variables ────────────────────────────────────

# Apply a log(x+1) transformation to all toxin measurement columns.
mycdf <- mycdf %>%
  mutate(across(all_of(response_vars), ~ signif(log1p(.x), 4)))

# ── 1.8. Add Unique Identifier ───────────────────────────────────────────────

mycdf <- mycdf %>%
  mutate(Unique_ID = factor(paste0("sample_", seq_len(n())))) %>%
  relocate(Unique_ID)


# ---- 2. Impute Missing Predictor Data with MICE ----

# ── 2.1. Prepare Data for Imputation ─────────────────────────────────────────
# Separate predictors from responses.
predictor_vars <- setdiff(names(mycdf), response_vars)

# Identify columns that are already complete (no missing values).
no_missing_cols <- predictor_vars[colSums(is.na(mycdf[predictor_vars])) == 0]

# Isolate columns that require imputation.
pred_cols_to_impute <- setdiff(predictor_vars, no_missing_cols)

# Convert date columns to numeric for the imputation model.
predictor_df <- mycdf %>%
  select(all_of(pred_cols_to_impute)) %>%
  mutate(across(where(is.Date), as.numeric))

# ── 2.2. Configure and Run MICE ──────────────────────────────────────────────
# Define imputation methods: random forest for numeric/factors, logistic regression for binary.
meth <- make.method(predictor_df)
factor_cols <- names(predictor_df)[sapply(predictor_df, is.factor)]
bin_factors <- factor_cols[sapply(predictor_df[factor_cols], function(x) nlevels(x) == 2)]
multi_factor <- setdiff(factor_cols, bin_factors)

meth[sapply(predictor_df, is.numeric)] <- "rf"
meth[bin_factors] <- "logreg"
meth[multi_factor] <- "rf"

# Exclude ultra-sparse columns (e.g., >95% missing) from imputation.
sparse_cols <- names(predictor_df)[colMeans(is.na(predictor_df)) > 0.95]
meth[sparse_cols] <- ""

# Build a predictor matrix to guide the imputation process.
predM <- quickpred(predictor_df, mincor = 0.05, exclude = no_missing_cols)

# Run the MICE algorithm.
imp <- mice(
  predictor_df,
  m = 5, 
  method = meth,
  predictorMatrix = predM,
  ridge = 1e-5,
  remove.collinear = TRUE,
  remove.constant = TRUE,
  seed = 1701,
  printFlag = FALSE
)

# ── 2.3. Re-assemble Final Data Frame ────────────────────────────────────────
# Extract a completed dataset.
pred_imp <- complete(imp, 1)

# Restore original Date class to imputed date columns.
date_cols_imp <- names(pred_imp)[sapply(mycdf[names(pred_imp)], inherits, "Date")]
pred_imp <- pred_imp %>%
  mutate(across(all_of(date_cols_imp), ~ as.Date(.x, origin = "1970-01-01")))

# Drop any columns that still contain NAs or are manually specified as problematic.
linked_cols <- c(
  "Insecticide_product", "Insecticide_application_time",
  "Fungicide_product_3", "Fungicide_dose_3",
  "Growth_regulator_product_1", "Growth_regulator_product_2"
)
leftover_na_cols <- names(pred_imp)[colSums(is.na(pred_imp)) > 0]
drop_cols <- union(leftover_na_cols, intersect(linked_cols, names(pred_imp)))

pred_imp <- pred_imp %>% select(-all_of(drop_cols))

# Combine the complete columns, imputed columns, and response variables.
mycdf <- bind_cols(
  mycdf[no_missing_cols],
  pred_imp,
  mycdf[response_vars]
)


# ---- 3. Handle Replicates and Reshape Data ----

# The dataset contains replicate measurements in columns like 'DON_rep1', 'DON_rep2'.
# This section pivots the data to a long format and then back to a wide format,
# creating a 'replicate_number' column to track each measurement.

mycdf_long <- mycdf %>%
  pivot_longer(
    cols = all_of(response_vars),
    names_to = "toxin_raw",
    values_to = "value"
  ) %>%
  mutate(
    # Extract replicate number from column name. Base measurements are '0'.
    replicate_number = case_when(
      str_detect(toxin_raw, "_rep\\d+$") ~ as.integer(str_extract(toxin_raw, "\\d+$")),
      str_detect(toxin_raw, "_rep$") ~ 1L,
      TRUE ~ 0L
    ),
    # Get the clean toxin name.
    toxin = str_remove(toxin_raw, "_rep\\d*$")
  ) %>%
  select(-toxin_raw)

# Pivot back to a wide format, with one row per sample per replicate.
mycdf <- mycdf_long %>%
  pivot_wider(
    names_from = toxin,
    values_from = value
  ) %>%
  mutate(replicate_number = factor(replicate_number))


# ---- 4. Weather Data Integration & Feature Engineering ----

# ── 4.1. Load and Prepare Weather Data ───────────────────────────────────────
# Load the master weather file containing daily data for multiple locations.
df_weather <- read.csv("weather_master.csv") %>%
  mutate(date = as.Date(date)) %>%
  select(lat, lon, date, rain, temp, rh) %>%
  distinct(lat, lon, date, .keep_all = TRUE) # Ensure no duplicate entries per location-day.

# Extract unique coordinates from the weather data for matching.
weather_coords <- df_weather %>% select(lat, lon) %>% distinct()

# ── 4.2. Match Samples to Nearest Weather Gridpoint ──────────────────────────

# Extract sample coordinates.
sample_coords <- mycdf %>% select(Unique_ID, Latitude, Longitude) %>% distinct()

# Use geosphere::distGeo to find the index of the nearest weather coordinate for each sample.
nn_indices <- sapply(1:nrow(sample_coords), function(i) {
  which.min(distGeo(
    p1 = c(sample_coords$Longitude[i], sample_coords$Latitude[i]),
    p2 = as.matrix(weather_coords[, c("lon", "lat")])
  ))
})

# Create a mapping from sample ID to the nearest weather coordinates.
sample_to_weather_map <- sample_coords %>%
  mutate(
    nearest_lat = weather_coords$lat[nn_indices],
    nearest_lon = weather_coords$lon[nn_indices]
  ) %>%
  select(Unique_ID, nearest_lat, nearest_lon)

# Join the nearest coordinates back to the main dataframe.
mycdf <- mycdf %>%
  left_join(sample_to_weather_map, by = "Unique_ID")

# ── 4.3. Engineer 90-Day Lagged Weather Features ─────────────────────────────

# Create a template by crossing each sample with a 90-day lag sequence.
lagged_tbl <- mycdf %>%
  select(Unique_ID, replicate_number, Harvest, nearest_lat, nearest_lon) %>%
  crossing(lag = 1:90) %>%
  # Calculate the specific calendar date for each lag day.
  mutate(weather_date = as.Date(Harvest) - days(lag)) %>%
  # Join the corresponding daily weather data.
  left_join(
    df_weather,
    by = c("nearest_lat" = "lat", "nearest_lon" = "lon", "weather_date" = "date")
  )

# Pivot the lagged data to a wide format, creating columns like 'rain_minus1', 'temp_minus90', etc.
mycdf_weather_features <- lagged_tbl %>%
  select(Unique_ID, replicate_number, lag, rain, temp, rh) %>%
  pivot_wider(
    names_from = lag,
    values_from = c(rain, temp, rh),
    names_glue = "{.value}_minus{lag}",
    names_sort = TRUE
  )

# Join the new weather features back to the main dataframe.
mycdf <- mycdf %>%
  left_join(mycdf_weather_features, by = c("Unique_ID", "replicate_number"))


# ---- 5. Final Feature Engineering & Cleaning ----

# ── 5.1. Create Binary Response Variables ────────────────────────────────────

# Define the continuous response variables.
cont_outcomes <- names(mycdf)[match("Trichothence_producer", names(mycdf)):match("replicate_number", names(mycdf)) - 1]

# Create new binary columns (e.g., 'DON_bin') where 1 indicates presence (>0) and 0 indicates absence.
mycdf <- mycdf %>%
  mutate(across(
    .cols = all_of(cont_outcomes),
    .fns = ~ as.integer(.x > 0),
    .names = "{.col}_bin"
  ))

# ── 5.2. Convert Date Predictors to Cyclical Features ────────────────────────

# Identify remaining date columns (predictors).
date_cols_to_transform <- mycdf %>% select(where(is.Date), -Harvest) %>% names()

# Convert day-of-year to sin/cos values to represent seasonality.
mycdf <- mycdf %>%
  mutate(across(
    .cols = all_of(date_cols_to_transform),
    .fns = list(
      yday_sin = ~ sin(2 * pi * yday(.x) / 365),
      yday_cos = ~ cos(2 * pi * yday(.x) / 365)
    ),
    .names = "{.col}_{.fn}"
  ))

# ── 5.3. Final Column Cleanup ────────────────────────────────────────────────

# Remove original date columns, location identifiers, and any other columns not needed for modeling.
id_cols_to_drop <- c(
  "Unique_ID", "Sample_code", "Institute", "Location_details", "Origin",
  "nearest_lat", "nearest_lon", "Harvest"
)

mycdf <- mycdf %>%
  select(
    -all_of(date_cols_to_transform),
    -all_of(id_cols_to_drop)
  ) %>%
  # Ensure replicate_number is numeric for modeling.
  mutate(replicate_number = as.numeric(as.character(replicate_number)))


# ---- 6. One-Hot Encode and Normalise for Modeling ----

# ── 6.1. Define Column Roles for 'recipes' ───────────────────────────────────
# Automatically identify predictor and outcome columns.
all_resp_cols <- mycdf %>% select(matches("Trichothence_producer|DON|NIV|ZEN|HT2|T2"),
                                  -ends_with(c("_sin", "_cos"))) %>% names()
bin_outcomes <- all_resp_cols[str_detect(all_resp_cols, "_bin$")]
cont_outcomes <- setdiff(all_resp_cols, bin_outcomes)

# ── 6.2. Create and Apply the Preprocessing Recipe ───────────────────────────
# The 'recipes' package provides a framework for structured data preprocessing.
rec <- recipe(~ ., data = mycdf) %>%
  # Assign roles to columns (predictor vs. outcome).
  update_role(all_of(c(cont_outcomes, bin_outcomes)), new_role = "outcome") %>%
  update_role(-all_of(c(cont_outcomes, bin_outcomes)), new_role = "predictor") %>%
  # Remove the 'Year' column and any zero-variance predictors.
  step_rm(Year) %>%
  step_zv(all_predictors()) %>%
  # One-hot encode all nominal (character/factor) predictors.
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Normalise numeric predictors, excluding binary flags and cyclical features.
  step_normalize(all_numeric_predictors(), -all_outcomes(), -contains("_bin"), -contains("_sin"), -contains("_cos"))

# Prepare and apply the recipe to the dataset.
prep_rec <- prep(rec, training = mycdf)
dat_nn <- bake(prep_rec, new_data = NULL)

# ── 6.3. Save the Final Model-Ready Dataset ──────────────────────────────────

# The resulting 'dat_nn' is a fully preprocessed, numeric data frame.
#write.csv(dat_nn, "clean_nn_data.csv", row.names = FALSE)
#saveRDS(dat_nn, "dat_nn.rds")
