################################################################################
# Mycotox-I: Model Performance Leaderboard & Visualization                    #
################################################################################
# This script loads and analyzes pre-computed performance metrics (R², RMSE,  #
# F1, AUC) for multiple machine learning models. It ranks the models for each #
# mycotoxin to determine the "winner" based on each metric.                  #
#                                                                            #
# The script produces two primary outputs:                                   #
#   1. A heatmap showing the winning model for each toxin/metric combination.  #
#   2. A bar chart summarizing the overall percentage of "wins" per model.     #
################################################################################


# ---- 0. Setup: Load Packages and Define Constants ----

library(tidyverse)
library(patchwork)

# Define mappings for cleaning names for display in plots.
CLEAN_TOXIN_NAMES <- c(
  "Trichothence_producer" = "Trichothecene Producer", "F_langsethiae" = "F. langsethiae",
  "F_poae" = "F. poae", "DON" = "DON", "D3G" = "D3G", "Nivalenol" = "Nivalenol",
  "3-AC-DON" = "3-AC-DON", "15-AC-DON" = "15-AC-DON", "T-2_toxin" = "T-2 Toxin",
  "HT-2_toxin" = "HT-2 Toxin", "T2G" = "T2G", "Neos" = "Neosolaniol",
  "ENN_A1" = "ENN A1", "ENN_A" = "ENN A", "ENN_B" = "ENN B", "ENN_B1" = "ENN B1",
  "BEAU" = "Beauvericin", "ZEN" = "Zearalenone", "Apicidin" = "Apicidin",
  "STER" = "Sterigmatocystin", "DAS" = "Diacetoxyscirpenol", "Quest" = "Questin",
  "AOH" = "Alternariol", "AME" = "Alternariol monomethyl ether", "MON" = "Moniliformin",
  "Ergocristine" = "Ergocristine", "EGT" = "Ergot alkaloids", "X3.AC.DON" = "3-AC-DON",
  "X15.AC.DON" = "15-AC-DON", "HT.2_toxin" = "HT-2 Toxin", "T.2_toxin" = "T-2 Toxin"
)

MODEL_MAP <- c(
  "OG_Model"          = "Baseline Neural Network",
  "kerasMLP_Freeze"   = "Transfer Learning (Frozen)",
  "kerasMLP_Unfreeze" = "Transfer Learning (Unfrozen)",
  "tabpfn"            = "TabPFN",
  "FTT"               = "FT-Transformer",
  "tabnet"            = "TabNet"
)

METRIC_MAP <- c("R2" = "R²", "RMSE" = "RMSE", "F1_or_Acc" = "F1", "AUC" = "AUC")

MODEL_COLORS <- c(
  "Baseline Neural Network"      = "lightgreen",
  "Transfer Learning (Frozen)"   = "purple",
  "Transfer Learning (Unfrozen)" = "brown",
  "TabPFN"                       = "lightyellow",
  "FT-Transformer"               = "lightblue",
  "TabNet"                       = "red"
)


# ---- 1. Load and Process Model Metrics ----
message("Loading and processing model metrics...")

# Load the saved metrics data.
all_metrics_scaled <- readRDS("/Users/alaninglis/Desktop/Transfer Learning models/Transfer Learning Results/all_metrics_scaled.rds")


# Pivot the data into a long format for easier ranking and analysis.
all_long <- all_metrics_scaled %>%
  pivot_longer(
    cols = c(RMSE, R2, F1_or_Acc, AUC),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    # Create an 'adjusted' value for ranking. Lower is better for RMSE,
    # while higher is better for others (so we negate them).
    adjusted_value = case_when(
      is.na(Value)    ~ Inf,
      Metric == "RMSE" ~ Value,
      TRUE             ~ -Value
    ),
    Metric = factor(Metric, levels = c("R2", "RMSE", "F1_or_Acc", "AUC"))
  )


# ---- 2. Rank Models and Generate Leaderboards ----
message("Ranking models and generating leaderboards...")

# Rank models for each toxin and metric combination.
ranks_tbl <- all_long %>%
  group_by(Toxin, Metric) %>%
  mutate(Rank = min_rank(adjusted_value)) %>%
  ungroup()

# Identify the winning model for each toxin-metric pair.
winners_tbl <- ranks_tbl %>%
  group_by(Toxin, Metric) %>%
  slice_min(order_by = adjusted_value, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(Toxin, Metric, Model, Value) %>%
  arrange(Metric, Toxin)


# ---- 3. Prepare Data for Plotting ----
message("Preparing data for plotting...")

# Apply the clean name mappings for toxins, models, and metrics.
winners_clean <- winners_tbl %>%
  mutate(
    Toxin  = recode(Toxin, !!!CLEAN_TOXIN_NAMES),
    Model  = recode(Model, !!!MODEL_MAP),
    Metric = recode(as.character(Metric), !!!METRIC_MAP),
    Metric = factor(Metric, levels = c("R²", "RMSE", "F1", "AUC")),
    Value  = round(Value, 2)
  )

# Determine the y-axis order for the heatmap based on the winning model for R².
# This groups toxins by their best-performing model on the R² metric.
r2_winners <- winners_clean %>%
  filter(Metric == "R²") %>%
  arrange(Model, desc(Value))

# Create the final ordered factor for the y-axis.
toxin_order <- unique(r2_winners$Toxin)

winners_plot_data <- winners_clean %>%
  mutate(Toxin = factor(Toxin, levels = rev(toxin_order)))


# ---- 4. Generate Heatmap of Winning Models ----
message("Generating heatmap of winning models...")

heatmap_plot <-  ggplot(winners_clean, aes(x = Metric, y = Toxin, fill = Model)) +
  geom_tile() +
  geom_text(aes(label = round(Value, 2)), size = 3) +
  scale_fill_manual(values = c(
    "Baseline Neural Network"      = "lightgreen",
    "Transfer Learning (Frozen)"   = "purple", 
    "Transfer Learning (Unfrozen)" = "brown", 
    "TabPFN"                       = "lightyellow", 
    "FT-Transformer"               = "lightblue", 
    "TabNet"                       = "red"  
  ), drop = FALSE) +
  labs(x = NULL, y = NULL, fill = "Model") +
  theme_bw(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    axis.text.x.bottom = element_text(size = 12),
    axis.text.y        = element_text(size = 10)
  )


# Supplementary section figure
heatmap_plot

# ---- 5. Generate Bar Chart of Overall Wins ----
message("Generating bar chart of overall model wins...")

# Count the total number of wins for each model across all metrics.
overall_wins <- winners_clean %>%
  count(Model) %>%
  mutate(Percent = round(100 * n / sum(n), 1)) %>%
  arrange(desc(Percent))

# Create the bar plot.
barchart_plot <- ggplot(overall_wins, aes(x = reorder(Model, Percent), y = Percent, fill = Model)) +
  geom_col(color = 'black') +
  coord_flip(ylim = c(0, max(overall_wins$Percent) * 1.1)) +
  scale_fill_manual(values = MODEL_COLORS, guide = "none") +
  labs(
    x = NULL,
    y = "Percentage of Total Wins Across All Metrics"
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor.x = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )


# Figure 10 in paper
print(barchart_plot)
