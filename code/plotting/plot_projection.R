
library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)
library(ggpubr)
library(stringr)
# library(argparse)

# Source the plotting functions
source("/home/rajd2/rds/hpc-work/snRNAseq_2026/code/age_plots.r")

# Parse arguments manually using base R
args_list <- commandArgs(trailingOnly = TRUE)

if (length(args_list) < 2) {
  stop("Usage: Rscript plot_projection.R --input <csv> --output <png> [--title <title>] [--color_var <var>] [--facet_var <var>]")
}

input_file <- NULL
output_file <- NULL
plot_title <- "Developmental expression of AHBA C3 in Excitatory Neurons"
color_var <- "dataset_source"
facet_var <- NULL

# Very basic argument parsing
for (i in seq_along(args_list)) {
  if (args_list[i] == "--input" && i < length(args_list)) {
    input_file <- args_list[i+1]
  } else if (args_list[i] == "--output" && i < length(args_list)) {
    output_file <- args_list[i+1]
  } else if (args_list[i] == "--title" && i < length(args_list)) {
    plot_title <- args_list[i+1]
  } else if (args_list[i] == "--color_var" && i < length(args_list)) {
    color_var <- args_list[i+1]
  } else if (args_list[i] == "--facet_var" && i < length(args_list)) {
    facet_var <- args_list[i+1]
  }
}

if (is.null(input_file) || is.null(output_file)) {
    stop("Both --input and --output arguments are required.")
}

# Load data
message(paste("Loading data from", input_file))
df <- read_csv(input_file)

# Clean/Process Data
message("Processing data...")

# Rename dataset_source to source if present (to match age_plots.r expectation)
if ("dataset_source" %in% names(df)) {
    message("Renaming dataset_source to source...")
    df <- df %>% rename(source = dataset_source)
}

# Set Age Range factor levels and Filter NA
df <- df %>%
  mutate(age_range = factor(Age_Range4, levels=c("Infancy", "Childhood", "Adolescence", "Adulthood"))) %>%
  filter(!is.na(age_range)) %>%  # Remove NA/Unknown age categories
  mutate(C = factor(C, levels=c('C3+', 'C3-')))

# Plot 1: Age Scatter
message(paste("Generating Plot 1: Age Scatter (color by", color_var, ")..."))
p_age <- df %>% plot_age(color_var = color_var)

if (!is.null(facet_var)) {
    p_age <- p_age + facet_wrap(vars(!!sym(facet_var)))
}

# Plot 2: Boxplots
message("Generating Plot 2: Boxplots...")
comparisons <- list(
    c('Adolescence', 'Adulthood'),
    c('Adolescence', 'Childhood')
)

df_box <- df %>% rename(network = C)

p_boxes <- df_box %>% plot_boxes(color_var = color_var) + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# Combine
message("Combining plots...")
p_final <- (p_age | p_boxes) + 
    plot_annotation(tag_levels='a', title=plot_title)

# Save
ggsave(output_file, plot=p_final, width=14, height=7, dpi=300)
message(paste("Saved to", output_file))
