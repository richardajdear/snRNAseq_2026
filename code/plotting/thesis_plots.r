# mamba install r-base r-ggplot2 r-scales r-patchwork r-paletteer r-tidyverse rpy2
library(ggplot2)
library(scales)
library(patchwork)
library(paletteer)
library(ggbeeswarm)
library(ggh4x)
library(ggpubr)
library(tidyverse)


plot_ahba_vs_pc1 <- function(df, size=.2, alpha=0.1, title='', inset=T) {
    df <- df %>%
    pivot_longer(cols=c('C3+', 'C3-'), names_to='C', values_to='C_score') %>%
    filter(Cell_Lineage != 'Glial_progenitors')
    
    plot <- df %>% 
    ggplot(aes(x=snRNAseq_PC1, y=C_score)) +
    facet_grid(C ~ ., switch='y', scales='free') +
    geom_point(aes(color=Cell_Lineage), size=size, stroke=.1, alpha=alpha) +
    guides(color=guide_legend(override.aes=list(size=3, alpha=0.5))) +
    xlab('snRNAseq PC1') +
    scale_color_paletteer_d("RColorBrewer::Set1") +
    stat_cor(aes(group=1, label=..r.label..), size=4) +
    theme_classic() + 
    theme(
        title = element_text(size=10),
        legend.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=10),
        strip.text.y.left = element_text(angle=0, size=10, vjust=0.5),
        strip.background = element_blank(),
        strip.placement = 'outside',
        plot.tag = element_text(size=12, vjust=3, face='bold')        
    ) +
    ggtitle(title)

    inset_l23 <- df %>% 
        filter(Cell_Type == 'L2-3', C == 'C3+') %>%
        ggplot(aes(x=snRNAseq_PC1, y=C_score)) +
        geom_point(aes(color=Cell_Lineage), size=size, stroke=.1, alpha=alpha) +
        guides(color=F) +
        scale_color_paletteer_d("RColorBrewer::Set1") +
        stat_cor(aes(group=1, label=..r.label..), size=3) +
        theme_classic() + 
        theme(
            title = element_text(size=8),
            legend.title = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank(),
            axis.title = element_blank()
        ) + 
        ggtitle('L23')

    inset_oligo <- df %>% 
        filter(Cell_Type == 'Oligos', C == 'C3-') %>%
        ggplot(aes(x=snRNAseq_PC1, y=C_score)) +
        geom_point(aes(color=Cell_Lineage), size=size, stroke=.1, alpha=alpha) +
        guides(color=F) +
        scale_color_manual(values = RColorBrewer::brewer.pal(n = 8, name = "Set1")[4]) +
        stat_cor(aes(group=1, label=..r.label..), size=3) +
        theme_classic() + 
        theme(
            title = element_text(size=8),
            legend.title = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank(),
            axis.title = element_blank()
        ) + 
        ggtitle('Oligo.')

    if (inset==F) {
        return(plot)
    } else return(
    plot + 
        inset_element(inset_l23, left=0.02, right=0.32, bottom=0.65, top=0.9, ignore_tag=T, clip=F) +
        inset_element(inset_oligo, left=0.7, right=1, bottom=0.25,  top=0.5, ignore_tag=T, clip=F)
    )
}




plot_age <- function(df, nrow=2, facet_colors=NULL, ylims=NULL) {
    ylim_max <- quantile(df$value, .9999)
    ylim_min <- quantile(df$value, .0001)

    p <- df %>%
    filter(value < ylim_max & value > ylim_min) %>% 
    ggplot(aes(x=Age_log2, y=value)) +
    facet_grid(C~., switch='y', scales='free') + 
    # geom_vline(xintercept=log2(c(9,25)+1),color='grey',size=0.5) +
    geom_point(aes(color=Cell_Type), size=.1, alpha=.3) +
    # geom_jitter(aes(color=Pseudotime_pct), size=.1, alpha=.3, width=.02) +
    # geom_jitter(aes(color=origin), size=.1, alpha=.3, width=.02) +
    geom_smooth(aes(group=1), se=F, guide=F, size=.5) + 
    # scale_color_paletteer_d("ggprism::plasma") +
    # scale_color_paletteer_c("grDevices::Plasma", 
    #     limits = c(0,100),
    #     breaks = c(0,25,50,75,100),
    #     name = 'Pseudotime (%)'
    # ) +
    scale_x_continuous(
        name = 'Donor Age',
        breaks = log2(1+c(0,1,9,25,60)),
        labels = function(x) round(2^x-1, 1)
    ) +
    scale_y_continuous(name='Expression (CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    scale_colour_viridis_d(option='viridis') +
    guides(color = guide_legend(byrow=T, override.aes = list(alpha=1, size=2))) +
    # guides(color = guide_colorbar(direction='horizontal', barwidth=5, barheight=1, theme=theme(title=element_text(size=12, vjust=.9)))) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=12, color='black'),
        axis.text.x = element_text(),
        # axis.title.x = element_text(size=12, color='black', vjust=2),
        panel.grid.major = element_line(size=.5), #element_blank(),
        # panel.grid.minor = element_blank(),
        legend.position='right',
        legend.title=element_blank(),
        legend.key.spacing.y = unit(2, 'mm'),
        strip.background = element_blank(),
        strip.text.y.left = element_text(size=12, angle=0),
        strip.text.x = element_text(size=12),
        # axis.title.y=element_blank(),
        # axis.ticks.y = element_blank(),
        # axis.text.y = element_blank(),
        axis.title.x = element_text(size=12, vjust=10)
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~C, nrow=nrow, scales='free_y', strip=strip)
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
    # geom_boxplot(aes(fill=after_stat(middle)), alpha=.3) +
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
    # scale_fill_viridis_d(option='inferno', begin=.1, end=.5) +
    # scale_fill_viridis_c(option='inferno', oob=squish) +
    # scale_fill_paletteer_c("grDevices::Viridis", oob=squish) +
    coord_cartesian(clip='off') +
    theme_classic() +
    theme(
        text = element_text(size=12, color='black'), 
        title = element_text(size=12, color='black'),
        # panel.grid = element_blank(),
        panel.grid.major = element_line(size=.5),
        panel.grid.minor = element_blank(),
        legend.position = 'none',
        strip.background = element_blank(),
        strip.text.x = element_text(size=12),
        strip.text.y.left = element_text(size=12, angle=0),
        # axis.ticks.y = element_blank(),
        # axis.text.y = element_blank(),
        axis.text.x = element_text(angle=30, hjust=1, size=10, color='black'),
        axis.title.x = element_blank()
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~network, nrow=nrow, scales=scales, strip=strip)
    } else {
        p <- p + facet_wrap(~network, nrow=nrow, scales=scales)
    }
    p
}




plot_age_combined <- function(df, nrow=2, colors=NULL, ylims=NULL, highlight=c(2,7)) {

    p <- df %>%
    mutate(network = factor(network, levels=unique(network), ordered=T)) %>%
    mutate(highlight = C %in% highlight) %>% 
    ggplot(aes(x=Age_log2, y=value)) +
    geom_smooth(aes(color=network, group=network, size=highlight), se=F) + 
    scale_size_manual(values=c(.5, 1.5), guide=F) +
    scale_x_continuous(
        name = 'Donor Age',
        breaks = log2(1+c(0,1,9,25,60)),
        labels = function(x) round(2^x-1, 1)
    ) +
    scale_y_continuous(name='Cell Expression (CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    scale_color_manual(values=colors) +
    guides(color = guide_legend(byrow=T, override.aes = list(alpha=1, size=2))) +
    theme_classic() +
    theme(
        text = element_text(size=12, color='black'),
        axis.text.x = element_text(),
        # axis.title.x = element_text(size=12, color='black', vjust=2),
        panel.grid.major = element_line(size=.5), #element_blank(),
        # panel.grid.minor = element_blank(),
        legend.position='right',
        legend.title=element_blank(),
        legend.key.spacing.y = unit(.5, 'mm'),
        strip.background = element_blank(),
        strip.text.y.left = element_text(size=12, angle=0),
        strip.text.x = element_text(size=12),
        # axis.title.y=element_blank(),
        # axis.ticks.y = element_blank(),
        # axis.text.y = element_blank(),
        axis.title.x = element_text(size=12)
    )

    p
}



plot_pseudotime_vs_age <- function(meta) {
    meta %>% 
    group_by(Individual, Cell_Type, Cell_Class) %>% 
    summarise(Age_log2 = mean(Age_log2), Pseudotime_pct = mean(Pseudotime_pct)) %>%
    ggplot(aes(x=Age_log2, y=Pseudotime_pct, color=Cell_Type)) +
    facet_wrap(~Cell_Class, ncol=3) +
    geom_point(size=0.5) +
    scale_color_paletteer_d("ggthemes::Tableau_20") +
    labs(x='Age (log2)', y='Pseudotime (%)', color='Cell Type') +
    theme_minimal() +
    theme(legend.position='bottom', legend.title=element_blank(),
          plot.title=element_text(hjust=0.5, size=12)) +
    ggtitle('Pseudotime vs Age')
}


plot_distributions <- function(meta, ylims=c(0,700000), color=Cell_Type) {
    meta %>% 
    arrange(age_range) %>%
    ggplot(aes(x=age_range, y=Pseudotime_pct, color={{ color }})) +
    # geom_jitter(size=.01, width=.3, alpha=.2) +
    geom_quasirandom(method = "tukeyDense", size=0.01, width=.3, alpha=.1) +
    scale_color_viridis_c(option='inferno', name='Expression\n(CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    # scale_color_paletteer_c("grDevices::Viridis", oob=squish) +
    # geom_boxplot(outlier.shape=NA, fill=NA, width=0.1, alpha=0.5, color='blue') +
    # geom_violin(fill=NA) +
    ylab('Cell Pseudotime (%)') +
    theme_classic() +
    theme(
        axis.text.x = element_text(size=10, angle=30, hjust=1, color='black'),
        axis.title.x = element_blank()
        # legend.position = 'none'
    )
}

plot_pseudotime <- function(df, nrow=2, facet_colors=NULL, ylims=ylims) {
    # ylim_max <- quantile(df$value, .99)
    # ylim_min <- quantile(df$value, .01)

    p <- df %>% 
    ggplot(aes(x=Pseudotime_pct, y=value, color=age_range)) +
    facet_grid(C~., switch='y', scales='free') + 
    geom_point(size=.01, alpha=.1) +
    # geom_smooth(aes(group=1)) +
    # stat_density_2d(geom='polygon', contour_var = "ndensity", color=NA, aes(group=age_range, fill=age_range, alpha=..level..)) +
    # scale_alpha_continuous(range = c(0.3,.7), guide=F) +
    scale_color_viridis_d(name="Age Range", option="mako") +
    guides(color = guide_legend(byrow=T, override.aes = list(alpha=1, size=2))) +
    # scale_fill_viridis_d(name="Age Range", option="mako") +
    scale_y_continuous(name='Expression (CPM)', limits=ylims, labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    xlab('Cell Pseudotime (%)') +
    # scale_color_paletteer_c("grDevices::Mako",
    #         breaks = log2(1+c(0,1,2,5,9,25,60)),
    #         labels = function(x) round(2^x-1, 1),
    #         name = 'Donor Age'
    #     ) +
    # guides(color = guide_colorbar(direction='horizontal', barwidth=5, barheight=1, theme=theme(title=element_text(size=12, vjust=.9)))) +
    theme_classic() +
    theme(
        # text = element_text(size=12, color='black'),
        axis.text.x = element_text(),
        # axis.title.x = element_text(size=12, color='black', vjust=2),
        # panel.grid.major = element_line(size=.5), #element_blank(),
        # panel.grid.minor = element_blank(),
        # legend.position='bottom',
        strip.text.x = element_blank(),
        strip.text.y.left = element_text(size=12, angle=0)
        # axis.title.y=element_blank(),
        # axis.text.y=element_blank()
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~C, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y')
    }
    p
}


plot_pseudotime_split <- function(networks_df, ylims=NULL) {

    df <- networks_df %>% 
    mutate(pseudotime_bin = cut(Pseudotime_pct, breaks=seq(0, 100, by=10)))

    df_line <- df %>% 
        group_by(pseudotime_bin, age_range) %>% 
        mutate(count=n()) %>% 
        filter(count > 10)

    p2 <- df %>% 
        ggplot(aes(x=pseudotime_bin, y=value, color=Cell_Type)) +
        facet_grid(C~age_range) +
        geom_quasirandom(method = "tukeyDense", size=0.01, width=.4, alpha=.5) +
        stat_summary(
            data = df_line,
            fun = mean,
            geom = 'line',
            aes(group = age_range),
            color='blue', size=.5,
            position = position_dodge(width = 0.85) 
        ) +
        scale_colour_viridis_d(name='Cell cluster', option='viridis') +
        guides(color = guide_legend(byrow=T, override.aes = list(alpha=1, size=2))) +
        # scale_color_paletteer_c("grDevices::Viridis", oob=squish, name='Expression\n(CPM)', 
        #     labels=function(y) paste0(round(y/1e3, 1), 'K')) +
        scale_y_continuous(name='Expression (CPM)', labels=function(y) paste0(round(y/1e3, 1), 'K')) +
        scale_x_discrete(
            name="Cell Pseudotime (%)",
            breaks=levels(df$pseudotime_bin)[c(T, F, F, F, F, T, F, F, F, T)],
            labels=c('0', '50', '100')
        ) +
        theme_classic() +
        theme(
            strip.text.y = element_blank(),
            # axis.title.y = element_blank(),
            axis.text.x = element_text(angle=0, hjust=.5),
            # axis.title.x = element_text(vjust=10),
            legend.key.spacing.y = unit(2, 'mm'),
            legend.position = 'right'
        )

    # comparisons = list(
    #     c('Adolescence', 'Childhood')
    #     # c('Adolescence', 'Adulthood')
    # )

    # p1 <- networks_df %>% 
    #     plot_boxes() + 
    #     stat_compare_means(comparisons = comparisons, color='blue', label='p.signif') + 
    #     scale_y_continuous(name='Expression (CPM)', labels=function(y) paste0(round(y/1e3, 1), 'K')) +
    #     theme_classic() +
    #     theme(
    #         strip.text.x = element_blank(),
    #         axis.text.x = element_text(angle=60, hjust=1, color='black'),
    #         axis.title.x = element_blank(),
    #         legend.position = 'none'
    #     ) + 
    #     ggtitle("Donors")

    if (!is.null(ylims)) {
        # p1 <- p1 + scale_y_continuous(limits=ylims, name='Expression (CPM)', labels=function(y) paste0(round(y/1e3, 1), 'K'))
        p2 <- p2 + scale_y_continuous(limits=ylims, name='Expression (CPM)', labels=function(y) paste0(round(y/1e3, 1), 'K'))
    }

    p2
    # p1 + p2 + plot_layout(widths=c(1,5))
}



plot_age_bins <- function(df, facet_colors=NULL, nrow=2) {
    p <- df %>% 
    arrange(Age_Range2) %>%
    mutate(
        Age_Range2 = str_replace(Age_Range2, '2nd trimester', '2nd tri.'),
        Age_Range2 = str_replace(Age_Range2, '3rd trimester', '3rd tri.'),
        Age_Range2 = factor(Age_Range2, ordered=T, levels=unique(Age_Range2))
    ) %>%
    ggplot(aes(x=Age_Range2, y=value)) + 
    facet_grid(C~., switch='y', scales='free_y') + 
    geom_jitter(aes(color=origin), size=.1, alpha=.3, width=.4) +
    # geom_boxplot(aes(fill=after_stat(middle)), alpha=.3) +
    xlab('Donor Age') +
    stat_summary(
        fun = median,
        geom = 'line',
        aes(group = 1), 
        color='blue', size=1,
        position = position_dodge(width = 0.85) 
    ) +
    # scale_color_paletteer_c("grDevices::Plasma", 
    #     limits = c(0,100),
    #     breaks = c(0,25,50,75,100),
    #     name = 'Pseudotime (%)'
    # ) +    
    theme_minimal() +
    theme(
        text = element_text(size=12, color='black'), 
        title = element_text(size=12, color='black'),
        # panel.grid = element_blank(),
        panel.grid.major = element_line(size=.5),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle=45, hjust=1, size=10, color='black'),
        # legend.position = 'none',
        strip.text.y.left = element_text(size=12, angle=0),
        # axis.title.y=element_blank(),
        axis.text.y=element_blank()
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~C, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y')
    }
    p
}


plot_grid <- function(df, nrow=2, cap=1.5, facet_colors=NULL) {
    df <- df %>% 
        mutate(pseudotime_bin = cut(Pseudotime_pct,
                            breaks=seq(0,100,by=10))
    ) %>% 
    group_by(C, age_range, pseudotime_bin) %>% 
    summarise(value = mean(value), n_cells = n()) %>% 
    group_by(C, age_range) %>% 
    mutate(pct_cells = n_cells / sum(n_cells)) %>% 
    group_by(C) %>% 
    mutate(value = (value - mean(value)) / sd(value))

    p <- df %>% 
    ggplot(aes(x=age_range, y=pseudotime_bin, z=value)) + 
    geom_point(aes(color=value, size=(pct_cells))) +
    scale_size_continuous(range=c(0.5, 8), name='% cells by age', labels=function(x) percent(x)) +
    scale_color_paletteer_c("grDevices::Viridis", limits=c(-cap,cap), oob=squish) +
    theme_classic() +
    theme(
        axis.text.x = element_text(angle=45, hjust=1),
        legend.position = 'right'
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~C, nrow=nrow, strip=strip)
    } else {
        p <- p + facet_wrap(~C, nrow=nrow)
    }
    p
}



plot_network_expression <- function(df, nrow=2, facet_colors=NULL) {
    p <- df %>% 
    ggplot(aes(x=value, y=Pseudotime_pct, color=value)) +
    facet_grid(C~., switch='y', scales='free') + 
    geom_point(size=.1, alpha=.3) +
    geom_smooth() +
    ylab('Pseudotime (%)') +
    xlab('Network expression') +
    scale_color_paletteer_c("grDevices::Viridis") +
    # ylim(ylim_min, ylim_max) +
    # scale_color_paletteer_c("grDevices::Mako",
    #         breaks = log2(1+c(0,1,2,5,9,25,60)),
    #         labels = function(x) round(2^x-1, 1),
    #         name = 'Donor Age'
    #     ) +
    guides(color = guide_colorbar(direction='horizontal', barwidth=5, barheight=1, theme=theme(title=element_text(size=12, vjust=.9)))) +
    theme_minimal() +
    theme(
        text = element_text(size=12, color='black'),
        axis.text.x = element_text(size=12, color='black'),
        # axis.title.x = element_text(size=12, color='black', vjust=2),
        panel.grid.major = element_line(size=.5), #element_blank(),
        panel.grid.minor = element_blank(),
        legend.position='bottom',
        strip.text.y.left = element_text(size=12, angle=0)
    )

    if (!is.null(facet_colors)) {
        strip = strip_themed(background_x = elem_list_rect(fill=facet_colors))
        p <- p + facet_wrap2(~C, nrow=nrow, scales='free_y', strip=strip)
    } else {
        p <- p + facet_wrap(~C, nrow=nrow, scales='free_y')
    }
    p
}




plot_TF_stats <- function(TF_stats) {
    df <- TF_stats %>%
    mutate(
        sig_ahba = q_ahba < 0.05,
        sig_gwas = q_gwas < 0.05,
        sig_color = case_when(
            sig_ahba & sig_gwas ~ 'Both',
            sig_ahba ~ 'AHBA',
            sig_gwas ~ 'GWAS',
            TRUE ~ 'None'
        ),
    ) %>% 
    mutate(
        sig_color = factor(sig_color, levels=c('None', 'AHBA', 'GWAS','Both'), ordered=T)
    )

    df %>%  
    ggplot(aes(x=z_gwas, y=z_ahba)) +
    facet_wrap(~enrichment, ncol=2) +
    geom_vline(xintercept=0, color='grey', size=.2) +
    geom_hline(yintercept=0, color='grey', size=.2) +
    geom_point(aes(size=n_genes, color=sig_color), alpha=.3) +
    geom_text(aes(label=Network), size=3, vjust=1, hjust=1,
              data=df %>% filter(sig_gwas)) +
    scale_color_manual(name='FDR-sig', values=c('grey', 'blue', 'red', 'green')) +
    scale_size_continuous(name='# genes') +
    xlab('GWAS enrichment (z-score)') +
    ylab('AHBA C3 enrichment (z-score)') +
    theme_classic() +
    theme(
        legend.position = 'right',
        text = element_text(size=14, color='black'),
        strip.text = element_text(size=14, color='black'),
        axis.text = element_text(size=10, color='black'),
        strip.background = element_blank(),
        plot.title = element_text(hjust=0, size=16)
    ) +
    ggtitle('Regulon enrichment in AHBA C3 and GWAS')
}


plot_AHBA_network_stats <- function(AHBA_network_stats) {
    colors <- c('#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff')

    AHBA_network_stats %>% 
    mutate(network = factor(label, ordered=T, levels=unique(.$label))) %>% 
    mutate(sig = case_when(
        q < 0.001 ~ '***',
        q < 0.01 ~ '**',
        q < 0.05 ~ '*',
        TRUE ~ ''
    )) %>% 
    mutate(hjust = ifelse(z>0, 0, 1)) %>% 
    ggplot(aes(y = factor(network), x = z, fill = factor(network))) + 
    facet_wrap(~C, ncol=3) + 
    geom_col() + 
    geom_text(aes(label=sig, hjust=hjust), vjust=.75, size=5) +
    scale_fill_manual(values = colors) + 
    guides(fill='none') +
    ylab('Regulon network') +
    coord_cartesian(clip='off') +
    xlab('Permutation test z-score') +
    theme_classic() + 
    theme(
        # axis.text.x = element_text(angle=45, hjust=1),
        strip.background = element_rect(size=.5),
        plot.margin = margin(unit(c(10, 20, 10, 10), "mm"))
    )
}

plot_GWAS_network_stats <- function(networks_magma_results) {
    colors <- c('#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff')

    networks_magma_results %>% 
    arrange(Network) %>% 
    mutate(network = factor(label, ordered=T, levels=unique(.$label))) %>% 
    mutate(sig = case_when(
        q < 0.001 ~ '***',
        q < 0.01 ~ '**',
        q < 0.05 ~ '*',
        TRUE ~ ''
    )) %>% 
    ggplot(aes(y = network, x = -log10(q), fill = network)) + 
    facet_wrap(~enrichment, ncol=2) + 
    geom_col() + 
    geom_text(aes(label=sig, hjust=0), vjust=.75, size=5) +
    scale_fill_manual(values = colors) + 
    guides(fill='none') +
    ylab('Regulon network') +
    coord_cartesian(clip='off') +
    xlab('-log10(FDR)') +
    theme_classic() + 
    theme(
        strip.background = element_rect(size=.5),
        panel.spacing = unit(5, "mm"),
        plot.margin = margin(unit(c(10, 20, 10, 10), "mm"))
    )
}