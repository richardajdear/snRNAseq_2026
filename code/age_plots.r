library(ggplot2)
library(scales)
library(patchwork)
# library(paletteer)
library(ggbeeswarm)
# library(ggh4x)
library(ggpubr)
library(tidyverse)

plot_age <- function(df, nrow=2, facet_colors=NULL, ylims=NULL) {
    ylim_max <- quantile(df$value, .9999)
    ylim_min <- quantile(df$value, .0001)

    p <- df %>%
    filter(value < ylim_max & value > ylim_min) %>% 
    ggplot(aes(x=Age_log2, y=value)) +
    facet_grid(C~., switch='y', scales='free') + 
    geom_point(aes(color=Cell_Type), size=.1, alpha=.3) +
    geom_smooth(aes(group=1), se=F, guide=F, size=.5) + 
    scale_x_continuous(
        name = 'Donor Age',
        breaks = log2(1+c(0,1,9,25,60)),
        labels = function(x) round(2^x-1, 1)
    ) +
    scale_y_continuous(name='Expression (CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    scale_colour_viridis_d(option='viridis') +
    guides(color = guide_legend(byrow=T, override.aes = list(alpha=1, size=2))) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=12, color='black'),
        axis.text.x = element_text(),
        panel.grid.major = element_line(size=.5), 
        legend.position='right',
        legend.title=element_blank(),
        legend.key.spacing.y = unit(2, 'mm'),
        strip.background = element_blank(),
        strip.text.y.left = element_text(size=12, angle=0),
        strip.text.x = element_text(size=12),
        axis.title.x = element_text(size=12, vjust=10)
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y', strip=strip) # facet_wrap2 needs ggh4x
    } else {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y')
    }
    p
}

plot_boxes <- function(df, facet_colors=NULL, nrow=2, scales='free_y', ylims=NULL, expand=c(.1,0)) {
    p <- df %>% 
    arrange(age_range) %>%
    mutate(
        age_range = str_replace(age_range, '2nd trimester', '2nd tri.'),
        age_range = str_replace(age_range, '3rd trimester', '3rd tri.'),
        age_range = factor(age_range, ordered=T, levels=unique(age_range))
    ) %>%
    group_by(network, Individual, age_range) %>% 
    summarize(value=mean(value)) %>% 
    mutate(highlight = ifelse(age_range == 'Adolescence', TRUE, FALSE)) %>%
    ggplot(aes(x=age_range, y=value)) + 
    geom_quasirandom(color='grey30') +
    geom_boxplot(aes(fill=highlight), alpha=.6, outlier.shape=NA) +
    xlab('Donor Age') +
    scale_y_continuous(name='Pseudobulked Expression (CPM)', limits=ylims, expand=expand, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=.5,
        position = position_dodge(width = 0.85) 
    ) +
    scale_fill_manual(values=c(NA,'firebrick2')) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=12, color='black'), 
        title = element_text(size=12, color='black'),
        panel.grid.major = element_line(size=.5),
        panel.grid.minor = element_blank(),
        legend.position = 'none',
        strip.background = element_blank(),
        strip.text.x = element_text(size=12),
        strip.text.y.left = element_text(size=12, angle=0),
        axis.text.x = element_text(angle=30, hjust=1, size=10, color='black'),
        axis.title.x = element_blank()
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap(~network, nrow=nrow, scales=scales, strip=strip) # facet_wrap2 needs ggh4x
    } else {
        p <- p + facet_wrap(~network, nrow=nrow, scales=scales)
    }
    p
}
