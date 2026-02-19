import os

rmd_path = 'notebooks/ahbaC3_projection/analysis.Rmd'
new_content = """### 2. Developmental Trends (Excitatory Neurons)

We use shared plotting functions to visualize the developmental trends.

```{r combined_analysis, fig.width=14, fig.height=7}
# Source shared plotting functions
# Path relative to notebooks/ahbaC3_projection/
source("../../code/age_plots.r")

# Prepare data for plotting functions
plot_data <- data %>%
  filter(region == 'prefrontal cortex') %>% # Ensure we are looking at PFC
  filter(str_detect(cell_class, "Excitatory") | str_detect(cell_type, "Excitatory")) %>%
  tidyr::pivot_longer(cols = c(`C3+`, `C3-`), names_to = "C", values_to = "value") %>%
  mutate(C = factor(C, levels=c('C3+', 'C3-'))) %>%
  mutate(age_range = case_when(
    age_category == "Prenatal" ~ "Prenatal",
    age_category == "Infant" ~ "Infancy",
    age_category == "Childhood" ~ "Childhood", 
    age_category == "Adolescence" ~ "Adolescence",
    age_category == "Adulthood" ~ "Adulthood",
    TRUE ~ NA_character_
  )) %>%
  mutate(age_range = factor(age_range, levels=c("Prenatal", "Infancy", "Childhood", "Adolescence", "Adulthood"))) %>%
  filter(!is.na(age_range)) %>%
  rename(Age_log2 = age_log2) %>% # Match case for plot_age
  rename(Individual = donor_id)   # Match case for plot_boxes

# Rename source if needed
if ("dataset_source" %in% names(plot_data) && !"source" %in% names(plot_data)) {
  plot_data <- plot_data %>% rename(source = dataset_source)
}

# Plot 1: Age Scatter
p_age <- plot_data %>% plot_age(color_var = "source")

# Plot 2: Boxplots
plot_data_box <- plot_data %>% rename(network = C)

comparisons <- list(
    c('Adolescence', 'Adulthood'),
    c('Adolescence', 'Childhood')
)

p_boxes <- plot_data_box %>% plot_boxes(color_var = "source") + 
  stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# Combine
(p_age | p_boxes) + 
  plot_annotation(tag_levels='a', title="Developmental expression of AHBA C3 in Excitatory Neurons") +
  plot_layout(guides = 'collect') & 
  theme(legend.position = 'right')
```

### 3. C3+ Score by Cell Type
"""

with open(rmd_path, 'r') as f:
    lines = f.readlines()

output_lines = []
found = False
for line in lines:
    if "### 2. C3+ Score vs Age (Excitatory Neurons)" in line:
        found = True
        break
    output_lines.append(line)

if found:
    with open(rmd_path, 'w') as f:
        f.writelines(output_lines)
        f.write(new_content)
        # Add back section 3 header and logic if I dropped it?
        # My new_content ends with "### 3. C3+ Score by Cell Type"
        # I need to append the rest of the file from where section 3 started.
        
    # Re-read to find where section 3 started in original
    # Wait, I stopped reading at Section 2.
    # I need to find where Section 3 starts and preserve it.
    
    # Improved logic:
    # 1. Keep lines until "### 2. C3+ Score vs Age..."
    # 2. Insert new_content (which does NOT include section 3 header)
    # 3. Skip lines until "### 3. C3+ Score by Cell Type"
    # 4. Keep lines from there to end.

    final_lines = output_lines[:]
    final_lines.append(new_content.replace("### 3. C3+ Score by Cell Type\n", "")) # Remove trailing header from new_content
    
    # Find start of section 3 in remaining lines
    rest = lines[len(output_lines):]
    sec3_idx = -1
    for i, line in enumerate(rest):
        if "### 3. C3+ Score by Cell Type" in line:
            sec3_idx = i
            break
    
    if sec3_idx != -1:
        final_lines.extend(rest[sec3_idx:])
    
    with open(rmd_path, 'w') as f:
        f.writelines(final_lines)
    print("Successfully updated Rmd.")
else:
    print("Target section not found.")

