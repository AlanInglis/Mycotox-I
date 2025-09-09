########################################################################
## Mycotox-I: detection rates, summaries, distributions, missingness  ##
##                                                                    ##
########################################################################
##   • compute per-toxin detection rates and summary statistics        ##
##   • visualise top-6 concentration distributions                     ##
##   • profile missingness across toxins                               ##
## Notes – Data are private; this script illustrates the EDA workflow. ##
########################################################################
## Script outline                                                      ##
##   0. Setup: packages, theme                                         ##
##   1. Load data and back-transform continuous toxins                 ##
##   2. Detection rates (Figure 5)                                     ##
##   3. Summary statistics table (Table 2)                             ##
##   4. Top-6 distributions (Figure 6)                                 ##
##   5. Missingness profile (Figure 7)                                 ##
########################################################################


# ============================== 0. Setup ===============================

# Packages (reduced to those actually used)
library(ggplot2)
library(dplyr)
library(tidyr)
library(VIM)

theme_set(theme_bw())


# ==================== 1. Load data & back-transform ===================

mycdf <- readRDS("mycdf_with_weather.rds")

start_resp <- match("Trichothence_producer", names(mycdf))
resp_cols  <- names(mycdf)[start_resp:length(mycdf)]
bin_vars   <- resp_cols[grepl("_bin$", resp_cols)]
cont_vars  <- setdiff(resp_cols, bin_vars)

# Back-transform continuous toxins from log1p to original scale
mycdf <- mycdf |>
  mutate(across(all_of(cont_vars), ~ expm1(.x)))

# Continuous mycotoxin columns
mycotoxin_cols <- c(
  "Trichothence_producer","F_langsethiae","F_poae","DON","D3G",
  "Nivalenol","3-AC-DON","15-AC-DON","T-2_toxin","HT-2_toxin",
  "T2G","Neos","ENN_A1","ENN_A","ENN_B","ENN_B1","BEAU",
  "ZEN","Apicidin","STER","DAS","Quest","AOH","AME","MON",
  "Ergocristine","EGT"
)

# Display names
clean_names <- c(
  "Trichothence_producer" = "Trichothecene Producer",
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
  "AME"  = "Alternariol monomethyl ether",
  "MON"  = "Moniliformin",
  "Ergocristine" = "Ergocristine",
  "EGT" = "Ergot alkaloids"
)


# =========================== 2. Detection rates =======================

detection_summary <- mycdf %>%
  select(all_of(mycotoxin_cols)) %>%
  summarise(across(
    everything(),
    ~ sum(. > 0, na.rm = TRUE) / sum(!is.na(.)) * 100
  )) %>%
  pivot_longer(cols = everything(),
               names_to = "mycotoxin",
               values_to = "detection_rate") %>%
  mutate(mycotoxin_clean = clean_names[mycotoxin]) %>%
  arrange(desc(detection_rate))

# Figure 5
p1 <- ggplot(detection_summary,
             aes(x = reorder(mycotoxin_clean, detection_rate),
                 y = detection_rate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(x = "Mycotoxin", y = "Detection Rate (%)") +
  theme(axis.text.y = element_text(size = 8))


# ===================== 3. Summary statistics table ====================

summary_stats <- mycdf %>%
  select(all_of(mycotoxin_cols)) %>%
  summarise(across(
    everything(),
    list(
      mean     = ~ mean(., na.rm = TRUE),
      sd       = ~ sd(., na.rm = TRUE),
      min      = ~ min(., na.rm = TRUE),
      max      = ~ max(., na.rm = TRUE),
      detected = ~ sum(. > 0, na.rm = TRUE),
      total    = ~ sum(!is.na(.))
    )
  )) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  separate(variable, into = c("mycotoxin", "statistic"), sep = "_(?=[^_]*$)") %>%
  pivot_wider(names_from = statistic, values_from = value) %>%
  mutate(
    detection_rate   = round(detected / total * 100, 1),
    mycotoxin_clean  = clean_names[mycotoxin],
    mean = round(mean, 2), sd = round(sd, 2),
    min  = round(min,  2), max = round(max, 2)
  ) %>%
  select(mycotoxin_clean, mean, sd, min, max, detected, total, detection_rate) %>%
  arrange(desc(detection_rate))

names(summary_stats) <- c("Mycotoxin", "Mean", "SD", "Min", "Max",
                          "Detected", "Total", "Detection Rate (%)")

summary_stats <- summary_stats |> arrange(Mycotoxin)

print("Summary Statistics Table:")
print(summary_stats)


# ========== 4. Distributions for top-6 detected mycotoxins ==========

top_mycotoxins <- detection_summary$mycotoxin[1:6]

mycdf_long <- mycdf %>%
  select(all_of(top_mycotoxins)) %>%
  pivot_longer(cols = everything(),
               names_to = "mycotoxin",
               values_to = "concentration") %>%
  filter(concentration > 0) %>%
  mutate(mycotoxin_clean = clean_names[mycotoxin])

# Figure 6
p2 <- ggplot(mycdf_long, aes(x = log1p(concentration))) +
  geom_histogram(bins = 20, fill = "darkblue") +
  facet_wrap(~ mycotoxin_clean, scales = "free", ncol = 3) +
  labs(x = expression(log(Concentration + 1)), y = "Count")


# ====================== 5. Missing data pattern =======================

missing_pattern <- mycdf %>%
  select(all_of(mycotoxin_cols)) %>%
  VIM::aggr(
    col = c("navyblue", "red"), numbers = TRUE, sortVars = TRUE,
    labels = names(.), cex.axis = 0.7, gap = 3,
    main = "Missing Data Patterns in Mycotoxin Measurements",
    plot = FALSE
  )

missing_df <- missing_pattern$missings %>%
  mutate(
    mycotoxin_clean = clean_names[Variable],
    Percentage = round(Count / nrow(mycdf) * 100, 1)
  )

# Figure 7
barp <- ggplot(missing_df,
               aes(x = reorder(mycotoxin_clean, Percentage), y = Percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Mycotoxin", y = "Percentage Missing (%)")