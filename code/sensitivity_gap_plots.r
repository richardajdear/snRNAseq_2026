# sensitivity_gap_plots.r — Gap-model sensitivity functions for AHBA C3
#
# Childhood and adolescence are separated by a gap to account for uncertainty
# about when children enter adolescence.
#
# Requires: hvg_plots.r loaded first (for add_hvg_columns, condition_levels)

# ── Gap-model Sensitivity Computation ────────────────────────────────────────

compute_sensitivity_gap <- function(df, selected_conds,
                                     child_start = 1,
                                     child_ends = c(6, 7, 8, 9),
                                     adol_starts = c(12, 13, 14, 15),
                                     adol_ends = c(22, 23, 24, 25)) {
  params <- expand.grid(
    child_end  = child_ends,
    adol_start = adol_starts,
    adol_end   = adol_ends,
    stringsAsFactors = FALSE
  )

  all_stats <- list()
  for (i in seq_len(nrow(params))) {
    ce      <- params$child_end[i]
    adol_s  <- params$adol_start[i]
    ae      <- params$adol_end[i]

    pb <- df %>%
      filter(condition %in% selected_conds, C == 'C3+') %>%
      mutate(age_range = case_when(
        age_years >= child_start & age_years < ce     ~ "Childhood",
        age_years >= adol_s      & age_years < ae     ~ "Adolescence",
        TRUE ~ NA_character_
      )) %>%
      filter(!is.na(age_range)) %>%
      group_by(condition, individual, age_range) %>%
      summarize(value = mean(value), .groups = 'drop')

    tmp <- pb %>%
      group_by(condition) %>%
      summarize(
        n_child    = sum(age_range == 'Childhood'),
        n_adol     = sum(age_range == 'Adolescence'),
        mean_child = mean(value[age_range == 'Childhood']),
        mean_adol  = mean(value[age_range == 'Adolescence']),
        sd_child   = sd(value[age_range == 'Childhood']),
        sd_adol    = sd(value[age_range == 'Adolescence']),
        p_value = tryCatch(
          wilcox.test(value[age_range == 'Childhood'],
                      value[age_range == 'Adolescence'])$p.value,
          error = function(e) NA_real_
        ),
        .groups = 'drop'
      ) %>%
      mutate(
        pooled_sd = sqrt(((pmax(n_child, 1) - 1) * sd_child^2 +
                          (pmax(n_adol, 1) - 1) * sd_adol^2) /
                         pmax(n_child + n_adol - 2, 1)),
        cohens_d = ifelse(pooled_sd > 0,
                          (mean_child - mean_adol) / pooled_sd, NA_real_),
        min_detectable_d = mapply(function(n1, n2) {
          if (n1 < 2 || n2 < 2) return(NA_real_)
          tryCatch(
            pwr.t2n.test(n1 = n1, n2 = n2, power = 0.80,
                         sig.level = 0.05)$d,
            error = function(e) NA_real_
          )
        }, n_child, n_adol),
        child_end = ce, adol_start = adol_s, adol_end = ae
      )
    all_stats[[i]] <- tmp
  }

  bind_rows(all_stats) %>%
    mutate(
      condition = factor(condition, levels = selected_conds),
      log10p = -log10(p_value),
      signif = p_value < 0.05,
      p_star = case_when(
        p_value < 0.001 ~ '***',
        p_value < 0.01  ~ '**',
        p_value < 0.05  ~ '*',
        TRUE ~ 'ns'
      ),
      delta_k = (mean_child - mean_adol) / 1e3,
      d_label = paste0(sprintf("%.1f", delta_k), 'K\n', sprintf("%.2f", cohens_d)),
      p_label = ifelse(signif,
                       paste0(p_star, '\n', sprintf("%.2f", round(p_value, 2))),
                       ''),
      n_label = paste0(n_child, '/', n_adol, '\n',
                       sprintf("%.2f", round(min_detectable_d, 2)))
    )
}

select_best_gap <- function(sens_all, baseline = 'all_genes') {
  best <- sens_all %>%
    filter(condition == baseline) %>%
    arrange(p_value) %>%
    slice(1)
  list(child_end  = best$child_end,
       adol_start = best$adol_start,
       adol_end   = best$adol_end,
       p_value    = best$p_value,
       cohens_d   = best$cohens_d)
}

# ── Gap-model Sensitivity Heatmaps ──────────────────────────────────────────

plot_gap_cohens_d <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(adol_start), y = condition, fill = cohens_d)) +
    facet_grid(
      paste0("child < ", child_end, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = d_label), size = 1.7, lineheight = 0.85) +
    scale_fill_gradient2(
      low = '#2166AC', mid = 'white', high = '#B2182B', midpoint = 0,
      name = "Cohen's d"
    ) +
    labs(
      x = 'Adolescence start (years)',
      y = 'HVG condition',
      title = "C3+ effect size (Cohen's d): Childhood vs Adolescence",
      subtitle = "Childhood [1, upper), gap, Adolescence [lower, upper). Pseudobulked by individual."
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

plot_gap_pvalue <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(adol_start), y = condition, fill = log10p)) +
    facet_grid(
      paste0("child < ", child_end, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = p_label,
                  fontface = ifelse(signif, 'bold', 'plain')),
              size = 1.7, color = 'white') +
    scale_fill_gradient(
      low = 'grey90', high = '#B2182B',
      name = expression(-log[10](p))
    ) +
    labs(
      x = 'Adolescence start (years)',
      y = 'HVG condition',
      title = 'C3+ Wilcoxon p-value sensitivity to age range definitions',
      subtitle = 'Bold: p < 0.05. Childhood [1, upper), gap, Adolescence [lower, upper).'
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

plot_gap_power <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(adol_start), y = condition, fill = min_detectable_d)) +
    facet_grid(
      paste0("child < ", child_end, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = n_label), size = 1.7) +
    scale_fill_gradient(
      low = '#1A9850', high = 'grey95',
      name = "Min. detectable d"
    ) +
    labs(
      x = 'Adolescence start (years)',
      y = 'HVG condition',
      title = "Minimum detectable effect size at 80% power (Cohen's d, alpha = 0.05)",
      subtitle = "Cell labels: n_childhood / n_adolescence donors. Lower = more sensitive."
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

# ── Gap-model Trajectory & Box Plots ────────────────────────────────────────

plot_gap_trajectories <- function(df, child_start, child_end, adol_start, adol_end,
                                   zscore = FALSE) {
  df_plot <- df %>%
    filter(C == 'C3+', condition != 'all_genes') %>%
    add_hvg_columns()

  ylim_max <- quantile(df_plot$value, .999)
  ylim_min <- quantile(df_plot$value, .001)
  df_plot <- df_plot %>% filter(value >= ylim_min & value <= ylim_max)

  if (zscore) {
    df_plot <- df_plot %>%
      group_by(condition) %>%
      mutate(value = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE)) %>%
      ungroup()
    scales_arg <- 'free'
    y_scale <- scale_y_continuous(name = 'Z-score')
  } else {
    scales_arg <- 'free'
    y_scale <- scale_y_continuous(
      name = 'C3+ Score (CPM)',
      labels = function(y) paste0(round(y / 1e3, 1), 'K')
    )
  }

  df_plot %>%
    ggplot(aes(x = age_years, y = value)) +
    facet_grid(factor(n_genes) ~ flavor, scales = scales_arg) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(aes(color = source), size = 0.5, stroke = NA, alpha = 0.15) +
    geom_smooth(aes(group = 1), se = FALSE, color = 'black', linewidth = 0.5) +
    geom_vline(xintercept = child_start, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    geom_vline(xintercept = child_end, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = adol_start, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = adol_end, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    scale_x_continuous(
      name = 'Donor Age (years)',
      limits = c(0, 30),
      breaks = seq(0, 30, 5)
    ) +
    y_scale +
    scale_color_discrete(name = 'Source',
      guide = guide_legend(override.aes = list(size = 2, alpha = 1))) +
    theme_classic() +
    theme(
      text = element_text(size = 8),
      strip.text = element_text(size = 7),
      strip.text.y.right = element_text(angle = 0),
      axis.text.x = element_text(size = 6),
      panel.spacing = unit(0.3, 'lines')
    ) +
    ggtitle(
      "C3+ age trajectories by HVG method and n_top_genes",
      subtitle = sprintf(
        "Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)",
        child_start, child_end, child_end, adol_start, adol_start, adol_end))
}

plot_gap_pseudobulk <- function(df, child_start, child_end, adol_start, adol_end,
                                 zscore = FALSE) {
  pb <- df %>%
    filter(C == 'C3+', condition != 'all_genes') %>%
    add_hvg_columns() %>%
    group_by(condition, flavor, n_genes, individual, age_years, source) %>%
    summarize(value = mean(value), .groups = 'drop')

  if (zscore) {
    pb <- pb %>%
      group_by(condition) %>%
      mutate(value = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE)) %>%
      ungroup()
    scales_arg <- 'fixed'
    y_scale <- scale_y_continuous(name = 'Z-score')
  } else {
    scales_arg <- 'free'
    y_scale <- scale_y_continuous(
      name = 'C3+ Pseudobulked Score (CPM)',
      labels = function(y) paste0(round(y / 1e3, 1), 'K')
    )
  }

  pb %>%
    ggplot(aes(x = age_years, y = value)) +
    facet_grid(factor(n_genes) ~ flavor, scales = scales_arg) +
    annotate('rect', xmin = child_end, xmax = adol_start,
             ymin = -Inf, ymax = Inf, fill = 'grey80', alpha = 0.3) +
    geom_point(aes(color = source), size = .3, alpha = 0.5) +
    geom_smooth(aes(group = 1), se = FALSE, color = 'black', linewidth = 0.5) +
    geom_vline(xintercept = child_start, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    geom_vline(xintercept = child_end, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = adol_start, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = adol_end, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    scale_x_continuous(
      name = 'Donor Age (years)',
      limits = c(0, 30),
      breaks = seq(0, 30, 5)
    ) +
    y_scale +
    scale_color_discrete(name = 'Source',
      guide = guide_legend(override.aes = list(size = 2, alpha = 1))) +
    theme_classic() +
    theme(
      text = element_text(size = 8),
      strip.text = element_text(size = 7),
      strip.text.y.right = element_text(angle = 0),
      axis.text.x = element_text(size = 6),
      panel.spacing = unit(0.3, 'lines')
    ) +
    ggtitle(
      "C3+ pseudobulked age trajectories by HVG method",
      subtitle = sprintf(
        "Childhood [%d, %d), gap [%d, %d), Adolescence [%d, %d)",
        child_start, child_end, child_end, adol_start, adol_start, adol_end))
}

make_boxes_gap_df <- function(df, child_start, child_end, adol_start, adol_end) {
  df %>%
    rename(Individual = individual, network = C) %>%
    mutate(age_range = case_when(
      age_years < 0                                        ~ "Prenatal",
      age_years >= 0          & age_years < child_start    ~ "Infancy",
      age_years >= child_start & age_years < child_end     ~ "Childhood",
      age_years >= child_end  & age_years < adol_start     ~ "Gap",
      age_years >= adol_start & age_years < adol_end       ~ "Adolescence",
      age_years >= adol_end                                ~ "Adulthood",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(age_range)) %>%
    mutate(age_range = factor(age_range, ordered = TRUE,
                              levels = c("Prenatal", "Infancy", "Childhood",
                                         "Gap", "Adolescence", "Adulthood"))) %>%
    group_by(condition, network, Individual, age_range, source, age_years) %>%
    summarize(value = mean(value), .groups = 'drop')
}

plot_gap_boxes <- function(df_boxes, child_start, child_end, adol_start, adol_end,
                            zscore = FALSE) {
  comparisons <- list(c('Childhood', 'Adolescence'), c('Adolescence', 'Adulthood'))

  df_plot <- df_boxes %>%
    filter(network == 'C3+', condition != 'all_genes') %>%
    add_hvg_columns()

  if (zscore) {
    df_plot <- df_plot %>%
      group_by(condition) %>%
      mutate(value = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE)) %>%
      ungroup()
    scales_arg <- 'free_x'
    y_scale <- scale_y_continuous(name = 'Z-score')
  } else {
    scales_arg <- 'free'
    y_scale <- scale_y_continuous(
      name = 'Pseudobulked C3+ Score (CPM)',
      labels = function(y) paste0(round(y / 1e3, 1), 'K')
    )
  }

  df_plot %>%
    ggplot(aes(x = age_range, y = value)) +
    facet_grid(factor(n_genes) ~ flavor, scales = scales_arg) +
    geom_boxplot(outlier.shape = NA, alpha = 0.4) +
    geom_jitter(aes(color = source), width = 0.15, size = 0.5, alpha = 0.6) +
    stat_compare_means(comparisons = comparisons, label = 'p.signif', size = 3) +
    y_scale +
    scale_color_discrete(name = 'Source',
      guide = guide_legend(override.aes = list(size = 2, alpha = 1))) +
    coord_cartesian(clip = 'off') +
    theme_classic() +
    theme(
      text = element_text(size = 8),
      strip.text = element_text(size = 7),
      strip.text.y.right = element_text(angle = 0),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
      axis.title.x = element_blank(),
      panel.spacing.x = unit(0.3, 'lines'),
      panel.spacing.y = unit(1.5, 'lines'),
      panel.margin = margin(t = 15, r = 5, b = 5, l = 5)
    ) +
    ggtitle(sprintf(
      "C3+ Childhood [%d, %d) vs Adolescence [%d, %d) by HVG method",
      child_start, child_end, adol_start, adol_end))
}
