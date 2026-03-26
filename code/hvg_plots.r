# hvg_plots.r — Shared plotting functions for AHBA C3 HVG investigation
#
# Requires: ggplot2, dplyr, tidyr, patchwork, ggpubr, pwr
# Source this file after loading libraries.

# ── Helpers ──────────────────────────────────────────────────────────────────

condition_levels <- function(n_values) {
  c('all_genes',
    paste0('seurat_v3_', n_values),
    paste0('seurat_', n_values),
    paste0('pearson_', n_values))
}

prepare_r_data <- function(final_df, n_values) {
  final_df %>%
    mutate(Age_log2 = log2(age_years + 1)) %>%
    mutate(condition = factor(condition, levels = condition_levels(n_values)))
}

add_hvg_columns <- function(df, condition_col = 'condition') {
  df %>%
    mutate(
      flavor = case_when(
        grepl('seurat_v3', .data[[condition_col]]) ~ 'seurat_v3',
        grepl('pearson', .data[[condition_col]]) ~ 'pearson',
        grepl('seurat', .data[[condition_col]]) ~ 'seurat',
        TRUE ~ 'none'
      ),
      n_genes = ifelse(.data[[condition_col]] == 'all_genes', NA_real_,
                        as.numeric(gsub('.*_(\\d+)$', '\\1',
                                        as.character(.data[[condition_col]]))))
    )
}

# ── Part 1: Age Range Sensitivity ────────────────────────────────────────────

compute_sensitivity <- function(df, selected_conds,
                                child_starts = c(0.5, 1, 2, 3),
                                boundaries = seq(8, 14),
                                adol_ends = c(18, 21, 23, 25)) {
  params <- expand.grid(
    child_start = child_starts,
    boundary    = boundaries,
    adol_end    = adol_ends,
    stringsAsFactors = FALSE
  )

  all_stats <- list()
  for (i in seq_len(nrow(params))) {
    cs <- params$child_start[i]
    bd <- params$boundary[i]
    ae <- params$adol_end[i]

    pb <- df %>%
      filter(condition %in% selected_conds, C == 'C3+') %>%
      mutate(age_range = case_when(
        age_years >= cs & age_years < bd ~ "Childhood",
        age_years >= bd & age_years < ae ~ "Adolescence",
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
        power = mapply(function(n1, n2, d) {
          if (is.na(d) || n1 < 2 || n2 < 2) return(NA_real_)
          tryCatch(
            pwr.t2n.test(n1 = n1, n2 = n2, d = abs(d),
                         sig.level = 0.05)$power,
            error = function(e) NA_real_
          )
        }, n_child, n_adol, cohens_d),
        child_start = cs, boundary = bd, adol_end = ae
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
                       sprintf("%.2f", round(power, 2)))
    )
}

select_best_age_range <- function(sens_all, baseline = 'all_genes',
                                  min_adol_end = 21) {
  best <- sens_all %>%
    filter(condition == baseline, adol_end >= min_adol_end) %>%
    arrange(p_value) %>%
    slice(1)
  list(child_start = best$child_start,
       boundary    = best$boundary,
       adol_end    = best$adol_end,
       p_value     = best$p_value,
       cohens_d    = best$cohens_d)
}

# ── Sensitivity Heatmaps ────────────────────────────────────────────────────

plot_sensitivity_cohens_d <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(boundary), y = condition, fill = cohens_d)) +
    facet_grid(
      paste0("child \u2265 ", child_start, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = d_label), size = 2, lineheight = 0.85) +
    scale_fill_gradient2(
      low = '#2166AC', mid = 'white', high = '#B2182B', midpoint = 0,
      name = "Cohen's d"
    ) +
    labs(
      x = 'Childhood / Adolescence boundary (years)',
      y = 'HVG condition',
      title = "C3+ effect size (Cohen's d): Childhood vs Adolescence",
      subtitle = 'd = (mean_child - mean_adol) / pooled_sd, pseudobulked by individual'
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

plot_sensitivity_pvalue <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(boundary), y = condition, fill = log10p)) +
    facet_grid(
      paste0("child \u2265 ", child_start, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = p_label,
                  fontface = ifelse(signif, 'bold', 'plain')),
              size = 2, color = 'white') +
    scale_fill_gradient(
      low = 'grey90', high = '#B2182B',
      name = expression(-log[10](p))
    ) +
    labs(
      x = 'Childhood / Adolescence boundary (years)',
      y = 'HVG condition',
      title = 'C3+ Wilcoxon p-value sensitivity to age range definitions',
      subtitle = 'Bold values: p < 0.05'
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

plot_sensitivity_power <- function(sens_all) {
  sens_all %>%
    ggplot(aes(x = factor(boundary), y = condition, fill = power)) +
    facet_grid(
      paste0("child \u2265 ", child_start, "y") ~
      paste0("adol < ", adol_end, "y")
    ) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = n_label), size = 2) +
    scale_fill_gradient(
      low = 'grey95', high = '#1A9850', limits = c(0, 1),
      name = 'Power'
    ) +
    labs(
      x = 'Childhood / Adolescence boundary (years)',
      y = 'HVG condition',
      title = "Power analysis: ability to detect observed C3+ effect at p < 0.05",
      subtitle = "Cell labels: n_childhood / n_adolescence donors"
    ) +
    theme_minimal(base_size = 9) +
    theme(
      strip.text = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      panel.spacing = unit(0.4, 'lines'),
      plot.title = element_text(size = 11)
    )
}

# ── Part 2: HVG Comparison Plots ────────────────────────────────────────────

plot_gene_retention <- function(stats_df, n_values) {
  stats_df <- stats_df %>%
    mutate(
      flavor = case_when(
        condition == 'all_genes' ~ 'none',
        grepl('seurat_v3', condition) ~ 'seurat_v3',
        grepl('pearson', condition) ~ 'pearson_residuals',
        TRUE ~ 'seurat'
      ),
      condition = factor(condition, levels = condition_levels(n_values))
    ) %>%
    filter(!is.na(condition))

  baseline_val <- stats_df %>%
    filter(condition == 'all_genes') %>%
    pull(n_grn_genes_used)

  ggplot(stats_df, aes(x = condition, y = n_grn_genes_used, fill = flavor)) +
    geom_col() +
    geom_hline(yintercept = baseline_val, linetype = 'dashed', color = 'red') +
    geom_text(aes(label = n_grn_genes_used), vjust = -0.3, size = 2.5) +
    scale_fill_brewer(palette = 'Set2') +
    labs(x = NULL, y = 'GRN genes used in projection',
         title = 'GRN gene retention by HVG condition') +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
}

plot_age_trajectories <- function(df, best_cs, best_bd, best_ae, zscore = FALSE) {
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
    scales_arg <- 'fixed'
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
    geom_point(aes(color = source), size = 0.5, stroke = NA, alpha = 0.15) +
    geom_smooth(aes(group = 1), se = FALSE, color = 'black', linewidth = 0.5) +
    geom_vline(xintercept = best_cs, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    geom_vline(xintercept = best_bd, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = best_ae, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    scale_x_continuous(
      name = 'Donor Age (years)',
      limits = c(0, 25),
      breaks = c(0, 5, 10, 15, 20, 25)
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
        "Vertical lines: childhood start (%.1fy), boundary (%dy), adolescence end (%dy)",
        best_cs, best_bd, best_ae))
}

plot_pseudobulk_trajectories <- function(df, best_cs, best_bd, best_ae, zscore = FALSE) {
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
    geom_point(aes(color = source), size = .3, alpha = 0.5) +
    geom_smooth(aes(group = 1), se = FALSE, color = 'black', linewidth = 0.5) +
    geom_vline(xintercept = best_cs, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    geom_vline(xintercept = best_bd, linetype = 'dashed',
               color = 'red', linewidth = 0.4) +
    geom_vline(xintercept = best_ae, linetype = 'dotted',
               color = 'blue', linewidth = 0.3) +
    scale_x_continuous(
      name = 'Donor Age (years)',
      limits = c(0, 25),
      breaks = c(0, 5, 10, 15, 20, 25)
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
        "Vertical lines: childhood start (%.1fy), boundary (%dy), adolescence end (%dy)",
        best_cs, best_bd, best_ae))
}

make_boxes_df <- function(df, best_cs, best_bd, best_ae) {
  df %>%
    rename(Individual = individual, network = C) %>%
    mutate(age_range = case_when(
      age_years < 0                              ~ "Prenatal",
      age_years >= 0   & age_years < best_cs    ~ "Infancy",
      age_years >= best_cs & age_years < best_bd ~ "Childhood",
      age_years >= best_bd & age_years < best_ae ~ "Adolescence",
      age_years >= best_ae                       ~ "Adulthood",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(age_range)) %>%
    mutate(age_range = factor(age_range, ordered = TRUE,
                              levels = c("Prenatal", "Infancy", "Childhood",
                                         "Adolescence", "Adulthood"))) %>%
    group_by(condition, network, Individual, age_range, source, age_years) %>%
    summarize(value = mean(value), .groups = 'drop')
}

plot_boxes <- function(df_boxes, best_cs, best_bd, best_ae, zscore = FALSE) {
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
      "C3+ Childhood [%.1f-%d) vs Adolescence [%d-%d) by HVG method",
      best_cs, best_bd, best_bd, best_ae))
}

plot_effect_summary <- function(df_boxes, group1 = 'Childhood', group2 = 'Adolescence') {
  summary_df <- df_boxes %>%
    filter(network == 'C3+') %>%
    group_by(condition) %>%
    summarize(
      median_g1 = median(value[age_range == group1]),
      median_g2 = median(value[age_range == group2]),
      mean_g1   = mean(value[age_range == group1]),
      mean_g2   = mean(value[age_range == group2]),
      sd_g1     = sd(value[age_range == group1]),
      sd_g2     = sd(value[age_range == group2]),
      n_g1      = sum(age_range == group1),
      n_g2      = sum(age_range == group2),
      p_value = tryCatch(
        wilcox.test(value[age_range == group1],
                    value[age_range == group2])$p.value,
        error = function(e) NA_real_
      ),
      .groups = 'drop'
    ) %>%
    mutate(
      delta = median_g1 - median_g2,
      pooled_sd = sqrt(((pmax(n_g1, 1) - 1) * sd_g1^2 +
                        (pmax(n_g2, 1) - 1) * sd_g2^2) /
                       pmax(n_g1 + n_g2 - 2, 1)),
      cohens_d = ifelse(pooled_sd > 0,
                        (mean_g1 - mean_g2) / pooled_sd, NA_real_),
      power = mapply(function(n1, n2, d) {
        if (is.na(d) || n1 < 2 || n2 < 2) return(NA_real_)
        tryCatch(
          pwr.t2n.test(n1 = n1, n2 = n2, d = abs(d),
                       sig.level = 0.05)$power,
          error = function(e) NA_real_
        )
      }, n_g1, n_g2, cohens_d),
      flavor = case_when(
        condition == 'all_genes' ~ 'none',
        grepl('seurat_v3', condition) ~ 'seurat_v3',
        grepl('pearson', condition) ~ 'pearson_residuals',
        TRUE ~ 'seurat'
      ),
      n_genes = ifelse(condition == 'all_genes', NA_real_,
                        as.numeric(gsub('.*_(\\d+)$', '\\1', condition)))
    )

  ref <- summary_df %>% filter(condition == 'all_genes')
  hvg_df <- summary_df %>% filter(condition != 'all_genes')
  x_breaks <- sort(unique(hvg_df$n_genes))

  common_theme <- theme_classic() +
    theme(
      axis.text.x = element_text(size = 8),
      legend.position = 'none'
    )

  p1 <- hvg_df %>%
    ggplot(aes(x = n_genes, y = delta, color = flavor, group = flavor)) +
    geom_hline(yintercept = 0, linetype = 'dashed', color = 'grey50') +
    geom_hline(yintercept = ref$delta, linetype = 'dotted', color = 'grey30') +
    geom_point(size = 2.5) +
    geom_line(linewidth = 0.5) +
    scale_x_continuous(breaks = x_breaks) +
    scale_color_brewer(palette = 'Set2') +
    labs(x = 'n_top_genes', y = 'Delta (median)') +
    common_theme

  p2 <- hvg_df %>%
    ggplot(aes(x = n_genes, y = cohens_d, color = flavor, group = flavor)) +
    geom_hline(yintercept = 0, linetype = 'dashed', color = 'grey50') +
    geom_hline(yintercept = ref$cohens_d, linetype = 'dotted', color = 'grey30') +
    geom_point(size = 2.5) +
    geom_line(linewidth = 0.5) +
    scale_x_continuous(breaks = x_breaks) +
    scale_color_brewer(palette = 'Set2') +
    labs(x = 'n_top_genes', y = "Cohen's d") +
    common_theme

  p3 <- hvg_df %>%
    ggplot(aes(x = n_genes, y = -log10(p_value), color = flavor, group = flavor)) +
    geom_hline(yintercept = -log10(0.05), linetype = 'dashed', color = 'red') +
    geom_hline(yintercept = -log10(ref$p_value), linetype = 'dotted', color = 'grey30') +
    geom_point(size = 2.5) +
    geom_line(linewidth = 0.5) +
    scale_x_continuous(breaks = x_breaks) +
    scale_color_brewer(palette = 'Set2') +
    labs(x = 'n_top_genes', y = expression(-log[10](p))) +
    common_theme

  p4 <- hvg_df %>%
    ggplot(aes(x = n_genes, y = power, color = flavor, group = flavor)) +
    geom_hline(yintercept = 0.8, linetype = 'dashed', color = 'red') +
    geom_hline(yintercept = ref$power, linetype = 'dotted', color = 'grey30') +
    geom_point(size = 2.5) +
    geom_line(linewidth = 0.5) +
    scale_x_continuous(breaks = x_breaks) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_brewer(palette = 'Set2') +
    labs(x = 'n_top_genes', y = 'Power') +
    theme_classic() +
    theme(axis.text.x = element_text(size = 8))

  (p1 | p2 | p3 | p4) +
    plot_layout(guides = 'collect') +
    plot_annotation(
      title = sprintf('C3+ %s vs %s: effect by HVG condition', group1, group2),
      subtitle = 'Dotted grey line = all_genes baseline. Red dashes: p=0.05 / power=0.8.',
      tag_levels = 'a'
    )
}

# ── Part 5: HVG Gene Set Overlap (Euler diagrams) ────────────────────────────

plot_hvg_euler <- function(hvg_df, n_show = c(2000, 4000, 8000)) {
  plots <- lapply(n_show, function(n) {
    sets <- list(
      seurat_v3 = hvg_df$gene[hvg_df$condition == paste0('seurat_v3_', n)],
      seurat    = hvg_df$gene[hvg_df$condition == paste0('seurat_',    n)],
      pearson   = hvg_df$gene[hvg_df$condition == paste0('pearson_',   n)]
    )
    sets <- sets[sapply(sets, length) > 0]
    if (length(sets) < 2) {
      return(ggplot() + ggtitle(paste0('n=', n, ': no data')) + theme_void())
    }
    tryCatch({
      e <- eulerr::euler(sets, shape = 'ellipse')
      p <- plot(e,
                quantities = list(fontsize = 8),
                labels     = list(fontsize = 9),
                fills      = list(fill = c('#66c2a5', '#fc8d62', '#8da0cb'),
                                  alpha = 0.5),
                main       = list(label = paste0('n_top = ', n), fontsize = 10))
      ggplotify::as.ggplot(p)
    }, error = function(err) {
      ggplot() +
        ggtitle(paste0('n=', n, ': ', conditionMessage(err))) +
        theme_void()
    })
  })

  wrap_plots(plots, nrow = 1) +
    plot_annotation(
      title    = 'HVG gene set overlap between selection methods',
      subtitle = 'Ellipse areas proportional to set sizes. Sets: seurat_v3, seurat, pearson_residuals.'
    )
}
