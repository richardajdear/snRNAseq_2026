library(ggplot2)
library(scales)
library(patchwork)
# library(paletteer)
library(ggbeeswarm)
# library(ggh4x)
library(ggpubr)
library(tidyverse)

plot_age <- function(df, color_var="source", nrow=2, facet_colors=NULL, ylims=NULL) {
    # Ensure source order if present
    if ("source" %in% names(df)) {
        df <- df %>%
            mutate(source = factor(source, levels = c("AGING", "HBCC", "VELMESHEV", "WANG")))
    }

    ylim_max <- quantile(df$value, .999)
    ylim_min <- quantile(df$value, .001)

    p <- df %>%
    filter(value <= ylim_max & value >= ylim_min) %>% 
    ggplot(aes(x=Age_log2, y=value)) +
    facet_grid(C~., switch='y', scales='free') + 
    geom_point(aes(color=.data[[color_var]]), size=.1, alpha=.3) +
    geom_smooth(aes(group=1), se=F, color="black", size=.5) + 
    scale_x_continuous(
        name = 'Donor Age',
        breaks = log2(1+c(0,1,9,25,60)),
        labels = function(x) round(2^x-1, 1)
    ) +
    scale_y_continuous(name='Expression (CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    guides(color = guide_legend(override.aes = list(alpha=1, size=1))) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=14, color='black'),
        axis.text.x = element_text(),
        panel.grid.major = element_line(size=.4), 
        legend.position='right',
        legend.title=element_blank(),
        strip.background = element_blank(),
        strip.text.y.left = element_text(size=14, angle=0),
        strip.text.x = element_text(size=14),
        axis.title.x = element_text(size=14, vjust=10)
    )

    if (!is.null(facet_colors)) {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y') 
    } else {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y')
    }
    p
}

plot_boxes <- function(df, color_var="source", facet_colors=NULL, nrow=2, scales='free_y', ylims=NULL, expand=c(.1,0)) {
    # Ensure source order if present and being used
    if ("source" %in% names(df)) {
        df <- df %>%
            mutate(source = factor(source, levels = c("AGING", "HBCC", "VELMESHEV", "WANG")))
    }

    # Prepare data
    df_sum <- df %>% 
    arrange(age_range) %>%
    mutate(
        age_range = factor(age_range, ordered=T, levels=c("Prenatal", "Infant", "Childhood", "Adolescence", "Adulthood"))
    ) %>%
    group_by(network, Individual, age_range, !!sym(color_var)) %>% 
    summarize(value=mean(value), .groups="drop") %>% 
    mutate(highlight = ifelse(age_range == 'Adolescence', TRUE, FALSE))
    
    p <- df_sum %>%
    ggplot(aes(x=age_range, y=value)) + 
    geom_quasirandom(aes(color=.data[[color_var]]), alpha=0.8, size=1) +
    geom_boxplot(aes(fill=highlight), alpha=.4, outlier.shape=NA) +
    xlab('Donor Age') +
    scale_y_continuous(name='Pseudobulked Expression (CPM)', limits=ylims, expand=expand, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=.5,
        position = position_dodge(width = 0.85) 
    ) +
    scale_fill_manual(values=c(NA,'firebrick2'), guide='none') +
    guides(color = guide_legend(override.aes = list(alpha=1, size=1))) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=14, color='black'), 
        title = element_text(size=14, color='black'),
        panel.grid.major = element_line(size=.4),
        panel.grid.minor = element_blank(),
        legend.position = 'right',
        legend.title = element_blank(),
        strip.background = element_blank(),
        strip.text.x = element_text(size=14),
        strip.text.y.left = element_text(size=14, angle=0),
        axis.text.x = element_text(angle=30, hjust=1, size=12, color='black'),
        axis.title.x = element_blank()
    )

    if (!is.null(facet_colors)) {
        p <- p + facet_wrap(~network, nrow=nrow, scales=scales)
    } else {
        p <- p + facet_wrap(~network, nrow=nrow, scales=scales)
    }
    p
}
