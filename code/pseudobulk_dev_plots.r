# pseudobulk_dev_plots.r — Developmental trend plots for pseudobulk C3 analysis
#
# Designed for use with by_cell_class.h5ad and excitatory_by_celltype.h5ad
# pseudobulk data analysed across multiple scANVI integration variants.
#
# Requires: ggplot2, dplyr, tidyr, patchwork, ggpubr, pwr
# Source after hvg_plots.r and sensitivity_gap_plots.r.

# ── Condition level helpers ───────────────────────────────────────────────────

pb_condition_levels <- function(n_values) {
  # 'all_genes' baseline + pearson-only HVG conditions
  c('all_genes', paste0('pearson_', n_values))
}

prepare_pb_r_data <- function(final_df, n_values) {
  final_df %>%
    mutate(Age_log2 = log2(age_years + 1)) %>%
    mutate(condition = factor(condition, levels = pb_condition_levels(n_values)))
}

# ── Cell-Class Trajectory Plots ───────────────────────────────────────────────

#' Plot C3+ developmental trajectory faceted by cell_class.
#'
#' @param df         data.frame from prepare_pb_r_data (by_cell_class pseudobulk)
#' @param condition_label  one condition string, e.g. 'pearson_2000'
#' @param child_start,child_end,adol_start,adol_end  age range boundaries
#' @param color_var  obs column to colour points by (default 'integration')
#' @param zscore     if TRUE, z-score within each cell_class × integration group
plot_cellclass_trajectories <- function(
    df, condition_label,
    child_start, child_end, adol_start, adol_end,
    color_var = 'integration', zscore = FALSE) {

  df_plot <- df %>%
    filter(condition == condition_label, C == 'C3+', !is.na(cell_class)) %>%
    group_by(condition, individual, age_years, cell_class,
             .data[[color_var]]) %>%
    summarize(value = mean(value), .groups = 'drop')

  if (zscore) {
    df_plot <- df_plot %>%
      group_by(cell_class) %>%
      mutate(value = (value - mean(value, na.rm = TRUE)) /
               sd(value, na.rm = TRUE)) %>%
      ungroup()
    y_label <- 'Z-score (C3+)'
  } else {
    y_label <- 'C3+ Score (CPM)'
  }

  ggplot(df_plot, aes(x = age_years, y = value,
                      color = .data[[color_var]])) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(size = 0.7, alpha = 0.4) +
    geom_smooth(aes(group = .data[[color_var]]),
                se = FALSE, linewidth = 0.7, method = 'gam',
                formula = y ~ s(x, bs = 'cs')) +
    geom_vline(xintercept = c(child_start, child_end),
               linetype = c('dotted', 'dashed'),
               color = c('blue4', 'red3'), linewidth = 0.4) +
    geom_vline(xintercept = c(adol_start, adol_end),
               linetype = c('dashed', 'dotted'),
               color = c('red3', 'blue4'), linewidth = 0.4) +
    facet_wrap(~ cell_class, scales = 'free_y', nrow = 2) +
    scale_x_continuous(name = 'Donor Age (years)',
                       limits = c(-1, 30), breaks = seq(0, 30, 5)) +
    scale_y_continuous(name = y_label) +
    scale_color_brewer(palette = 'Set1', name = color_var) +
    theme_classic(base_size = 9) +
    theme(strip.text = element_text(size = 8),
          legend.position = 'bottom') +
    ggtitle(
      paste0('C3+ trajectories by cell class (', condition_label, ')'),
      subtitle = sprintf(
        'Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)',
        child_start, child_end, child_end, adol_start, adol_start, adol_end)
    )
}


#' Box plots of C3+ score by age range × cell_class.
plot_cellclass_boxes <- function(
    df, condition_label,
    child_start, child_end, adol_start, adol_end,
    color_var = 'integration') {

  df_plot <- df %>%
    filter(condition == condition_label, C == 'C3+', !is.na(cell_class)) %>%
    group_by(condition, individual, cell_class, .data[[color_var]], age_years) %>%
    summarize(value = mean(value), .groups = 'drop') %>%
    mutate(age_range = case_when(
      age_years <  0              ~ 'Prenatal',
      age_years <  child_start    ~ 'Infancy',
      age_years <  child_end      ~ 'Childhood',
      age_years <  adol_start     ~ 'Gap',
      age_years <  adol_end       ~ 'Adolescence',
      TRUE                        ~ 'Adulthood'
    )) %>%
    mutate(age_range = factor(age_range, ordered = TRUE,
      levels = c('Prenatal', 'Infancy', 'Childhood',
                 'Gap', 'Adolescence', 'Adulthood')))

  comparisons <- list(c('Childhood', 'Adolescence'),
                      c('Adolescence', 'Adulthood'))

  ggplot(df_plot, aes(x = age_range, y = value,
                      fill = .data[[color_var]])) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5,
                 position = position_dodge(0.8)) +
    geom_jitter(aes(color = .data[[color_var]]),
                position = position_jitterdodge(jitter.width = 0.15,
                                                dodge.width = 0.8),
                size = 0.5, alpha = 0.5) +
    facet_wrap(~ cell_class, scales = 'free_y', nrow = 2) +
    scale_x_discrete(name = NULL) +
    scale_y_continuous(name = 'C3+ Score (CPM)') +
    scale_fill_brewer(palette = 'Set1', name = color_var) +
    scale_color_brewer(palette = 'Set1', guide = 'none') +
    theme_classic(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          strip.text = element_text(size = 8),
          legend.position = 'bottom') +
    ggtitle(paste0('C3+ by age range and cell class (', condition_label, ')'))
}


# ── Cell-Type Aligned Trajectory Plots ────────────────────────────────────────

#' Plot C3+ trajectories faceted by cell_type_aligned.
#'
#' @param df              data.frame from prepare_pb_r_data (celltype pseudobulk)
#' @param condition_label one condition string
#' @param highlight_types optional character vector of cell types to highlight
plot_celltype_trajectories <- function(
    df, condition_label,
    child_start, child_end, adol_start, adol_end,
    highlight_types = NULL,
    color_var = 'integration', zscore = FALSE) {

  df_plot <- df %>%
    filter(condition == condition_label, C == 'C3+',
           !is.na(cell_type_aligned)) %>%
    group_by(condition, individual, age_years, cell_type_aligned,
             .data[[color_var]]) %>%
    summarize(value = mean(value), .groups = 'drop')

  if (!is.null(highlight_types)) {
    df_plot <- df_plot %>%
      filter(cell_type_aligned %in% highlight_types)
  }

  if (zscore) {
    df_plot <- df_plot %>%
      group_by(cell_type_aligned) %>%
      mutate(value = (value - mean(value, na.rm = TRUE)) /
               sd(value, na.rm = TRUE)) %>%
      ungroup()
    y_label <- 'Z-score (C3+)'
  } else {
    y_label <- 'C3+ Score (CPM)'
  }

  n_types <- length(unique(df_plot$cell_type_aligned))
  n_cols  <- ceiling(sqrt(n_types))

  ggplot(df_plot, aes(x = age_years, y = value,
                      color = .data[[color_var]])) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(size = 0.6, alpha = 0.4) +
    geom_smooth(aes(group = .data[[color_var]]),
                se = FALSE, linewidth = 0.7, method = 'gam',
                formula = y ~ s(x, bs = 'cs')) +
    geom_vline(xintercept = c(child_start, child_end),
               linetype = c('dotted', 'dashed'),
               color = c('blue4', 'red3'), linewidth = 0.4) +
    geom_vline(xintercept = c(adol_start, adol_end),
               linetype = c('dashed', 'dotted'),
               color = c('red3', 'blue4'), linewidth = 0.4) +
    facet_wrap(~ cell_type_aligned, scales = 'free_y', ncol = n_cols) +
    scale_x_continuous(name = 'Donor Age (years)',
                       limits = c(-1, 30), breaks = seq(0, 30, 5)) +
    scale_y_continuous(name = y_label) +
    scale_color_brewer(palette = 'Set1', name = color_var) +
    theme_classic(base_size = 9) +
    theme(strip.text = element_text(size = 7),
          axis.text.x = element_text(size = 7),
          legend.position = 'bottom') +
    ggtitle(
      paste0('C3+ trajectories by cell type (', condition_label, ')'),
      subtitle = sprintf(
        'Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)',
        child_start, child_end, child_end, adol_start, adol_start, adol_end)
    )
}


#' L2-3 focus: all conditions in one plot, coloured by cell_type_aligned.
plot_L23_focus <- function(
    df, child_start, child_end, adol_start, adol_end,
    L23_types = c('EN-L2_3-IT', 'EN-Newborn',
                  'EN-IT-Immature', 'EN-Non-IT-Immature'),
    color_var = 'cell_type_aligned') {

  df_plot <- df %>%
    filter(C == 'C3+', cell_type_aligned %in% L23_types) %>%
    group_by(condition, individual, age_years, .data[[color_var]]) %>%
    summarize(value = mean(value), .groups = 'drop')

  ggplot(df_plot, aes(x = age_years, y = value,
                      color = .data[[color_var]])) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(size = 0.7, alpha = 0.4) +
    geom_smooth(se = FALSE, linewidth = 0.8, method = 'loess',
                formula = y ~ x) +
    geom_vline(xintercept = c(child_start, child_end),
               linetype = c('dotted', 'dashed'),
               color = c('blue4', 'red3'), linewidth = 0.4) +
    geom_vline(xintercept = c(adol_start, adol_end),
               linetype = c('dashed', 'dotted'),
               color = c('red3', 'blue4'), linewidth = 0.4) +
    facet_wrap(~ condition, scales = 'free_y') +
    scale_x_continuous(name = 'Donor Age (years)',
                       limits = c(-1, 30), breaks = seq(0, 30, 5)) +
    scale_y_continuous(name = 'C3+ Score (CPM)') +
    scale_color_brewer(palette = 'Set2', name = 'Cell type') +
    theme_classic(base_size = 9) +
    theme(strip.text = element_text(size = 8),
          legend.position = 'bottom') +
    ggtitle('C3+ trajectories: L2-3 and EN subtypes',
            subtitle = sprintf(
              'Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)',
              child_start, child_end, child_end, adol_start,
              adol_start, adol_end))
}


#' L2-3 specific sensitivity grid (wraps compute_sensitivity_gap).
compute_L23_sensitivity <- function(
    df, selected_conds,
    L23_types = c('EN-L2_3-IT', 'EN-Newborn',
                  'EN-IT-Immature', 'EN-Non-IT-Immature'),
    child_start = 1,
    child_ends  = c(7, 8, 9, 10),
    adol_starts = c(10, 11, 12, 13, 14, 15),
    adol_ends   = c(19, 21, 23, 25)) {

  df_L23 <- df %>%
    filter(cell_type_aligned %in% L23_types)

  compute_sensitivity_gap(df_L23, selected_conds,
                          child_start = child_start,
                          child_ends  = child_ends,
                          adol_starts = adol_starts,
                          adol_ends   = adol_ends)
}


# ── Cross-Integration Comparison ─────────────────────────────────────────────

#' Compare sensitivity Cohen's d across integration variants for fixed age range.
plot_integration_cohens_d <- function(
    sens_list, condition_labels,
    child_end, adol_start, adol_end) {

  combined <- bind_rows(lapply(names(sens_list), function(nm) {
    sens_list[[nm]] %>% mutate(integration = nm)
  }))

  combined %>%
    filter(child_end == !!child_end,
           adol_start == !!adol_start,
           adol_end == !!adol_end,
           condition %in% condition_labels) %>%
    ggplot(aes(x = condition, y = cohens_d, fill = integration)) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    geom_hline(yintercept = 0, linewidth = 0.4) +
    scale_fill_brewer(palette = 'Set1', name = 'Integration') +
    scale_x_discrete(name = 'HVG condition') +
    scale_y_continuous(name = "Cohen's d (Childhood vs Adolescence)") +
    theme_classic(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
          legend.position = 'right') +
    ggtitle(
      sprintf("Cohen's d comparison across integrations"),
      subtitle = sprintf(
        'child < %dy, adol [%dy, %dy)',
        child_end, adol_start, adol_end))
}


#' Combined trajectory for a single cell type across integrations.
plot_celltype_cross_integration <- function(
    df_combined, cell_type_label, condition_label,
    child_start, child_end, adol_start, adol_end) {

  df_plot <- df_combined %>%
    filter(condition == condition_label, C == 'C3+',
           cell_type_aligned == cell_type_label) %>%
    group_by(integration, individual, age_years) %>%
    summarize(value = mean(value), .groups = 'drop')

  ggplot(df_plot, aes(x = age_years, y = value, color = integration)) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(size = 0.8, alpha = 0.4) +
    geom_smooth(se = TRUE, linewidth = 0.8, method = 'gam',
                formula = y ~ s(x, bs = 'cs'), alpha = 0.15) +
    geom_vline(xintercept = c(child_start, child_end),
               linetype = c('dotted', 'dashed'),
               color = c('blue4', 'red3'), linewidth = 0.4) +
    geom_vline(xintercept = c(adol_start, adol_end),
               linetype = c('dashed', 'dotted'),
               color = c('red3', 'blue4'), linewidth = 0.4) +
    scale_x_continuous(name = 'Donor Age (years)',
                       limits = c(-1, 30), breaks = seq(0, 30, 5)) +
    scale_y_continuous(name = 'C3+ Score (CPM)') +
    scale_color_brewer(palette = 'Set1', name = 'Integration') +
    theme_classic(base_size = 9) +
    theme(legend.position = 'right') +
    ggtitle(
      paste0('C3+ trajectory: ', cell_type_label,
             ' (', condition_label, ')'),
      subtitle = sprintf(
        'Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)',
        child_start, child_end, child_end, adol_start,
        adol_start, adol_end))
}
