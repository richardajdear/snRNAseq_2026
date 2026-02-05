library(ggplot2)
library(dplyr)
library(readr)
library(stringr)

# Settings
csv_path <- "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/combined_obs.csv"
out_donor_plot <- "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/donor_age_distribution.png"
out_cell_plot <- "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/cell_age_histogram.png"

# Load Data
cat("Loading CSV...\n")
df <- read_csv(csv_path, col_types = cols(.default = "c"))

# Function to clean "b'string'" format
clean_bytes <- function(x) {
  str_replace(x, "^b['\"](.*)['\"]$", "\\1")
}

# Clean columns
df <- df %>%
  mutate(
    source = clean_bytes(source),
    dataset = clean_bytes(dataset),
    individual = clean_bytes(individual),
    # age_years might be string due to mixed types in CSV warning earlier, ensure numeric
    age_years = as.numeric(clean_bytes(age_years))
  )

cat("Data loaded. Producing plot...\n")

# Process Data
# 1. Round age to integer
# 2. Distinct donors per Age and Source
plot_data <- df %>%
  mutate(age_round = round(age_years)) %>%
  filter(!is.na(age_round)) %>%
  group_by(age_round, source) %>%
  summarise(n_donors = n_distinct(individual), .groups = 'drop')

# Plot 1: Donor Age Distribution (Stacked)
p1 <- ggplot(plot_data, aes(x = age_round, y = n_donors, fill = source)) +
  geom_col() + 
  scale_x_continuous(breaks = seq(min(plot_data$age_round), max(plot_data$age_round), by = 5)) +
  labs(
    title = "Donor Age Distribution by Source",
    x = "Age (Years)",
    y = "Number of Unique Donors",
    fill = "Source"
  ) +
  theme_classic() + # White background
  theme(legend.position = "bottom")

# Save Donor Plot
ggsave(out_donor_plot, plot = p1, width = 10, height = 6, dpi = 300)
cat(paste("Donor plot saved to", out_donor_plot, "\n"))

# Plot 2: Cell Age Distribution (Histogram)
# We use the original df for this, ensuring age_years is numeric
# Rounding age_years to nearest integer as requested
cat("Generating cell histogram plot with rounded ages...\n")
p2 <- ggplot(df, aes(x = round(age_years), fill = source)) +
  geom_histogram(binwidth = 1, position = "stack") + 
  scale_x_continuous(breaks = seq(min(round(df$age_years), na.rm=TRUE), max(round(df$age_years), na.rm=TRUE), by = 5)) +
  labs(
    title = "Cell Age Distribution by Source (Stacked Histogram)",
    x = "Age (Years, rounded)",
    y = "Number of Cells",
    fill = "Source"
  ) +
  theme_classic() + # White background
  theme(legend.position = "bottom")

# Save Cell Plot
ggsave(out_cell_plot, plot = p2, width = 10, height = 6, dpi = 300)
cat(paste("Cell histogram plot saved to", out_cell_plot, "\n"))
