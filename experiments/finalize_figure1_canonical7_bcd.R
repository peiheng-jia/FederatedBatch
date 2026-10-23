#!/usr/bin/env Rscript

# Final manuscript rendering for Figure 1B-D from canonical-seven results.
# Figure 1A is intentionally excluded for manual preparation.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(cowplot)
})

source("external_artifacts/figure1/figures/figure1/r/color_config.R")

input_dir <- "results/figure1_canonical7_final_private"
output_dir <- "results/final_figure1_canonical7"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

raw_file <- file.path(input_dir, "figure1B_source_values_canonical7.csv")
umap_file <- file.path(input_dir, "figure1_umap_coordinates_final.csv")
tests_file <- file.path(input_dir, "paired_tests.csv")

canonical_seeds <- c(218L, 394L, 542L, 639L, 788L, 814L, 880L)
method_order <- c("PCA", "Harmony", "CONCORD", "FederatedBatch")
metric_order <- c(
  "ARI", "NMI", "ASW_batch", "iLISI", "ASW_cell", "Graph_connectivity"
)

raw <- read.csv(raw_file, check.names = FALSE)
required_raw <- c("method", "training_seed", "metric", "value")
if (!all(required_raw %in% names(raw))) {
  stop("Figure 1 source values are missing required columns")
}
if (!identical(sort(unique(raw$method)), sort(method_order))) {
  stop("Unexpected method names in Figure 1 source values")
}
for (method_name in method_order[-1]) {
  seeds <- raw %>%
    filter(.data$method == .env$method_name) %>%
    distinct(.data$training_seed) %>%
    pull(.data$training_seed) %>%
    as.integer() %>%
    sort()
  if (!identical(seeds, canonical_seeds)) {
    stop(sprintf("%s does not contain the canonical seven seeds", method_name))
  }
}

summary <- raw %>%
  mutate(
    Method = factor(.data$method, levels = method_order),
    Metric = factor(.data$metric, levels = metric_order)
  ) %>%
  group_by(.data$Method, .data$Metric) %>%
  summarise(
    Mean = mean(.data$value),
    SD = ifelse(dplyr::n() > 1, sd(.data$value), NA_real_),
    N = dplyr::n(),
    .groups = "drop"
  )

if (nrow(summary) != 24) {
  stop(sprintf("Expected 24 Figure 1 summary rows, found %d", nrow(summary)))
}

method_labels <- c(
  "PCA" = "PCA\n(n=1)",
  "Harmony" = "Harmony\n(n=7)",
  "CONCORD" = "CONCORD\n(n=7)",
  "FederatedBatch" = "FederatedBatch\n(n=7)"
)
method_colors <- c(
  "PCA" = METHOD_COLORS[["PCA"]],
  "Harmony" = METHOD_COLORS[["Harmony"]],
  "CONCORD" = METHOD_COLORS[["CONCORD"]],
  "FederatedBatch" = METHOD_COLORS[["FedCONCORD"]]
)

summary <- summary %>%
  mutate(
    Method_Label = factor(
      method_labels[as.character(.data$Method)],
      levels = unname(method_labels[method_order])
    )
  )

metric_theme <- theme_classic() +
  theme(
    axis.title = element_text(size = 7.5),
    axis.text.x = element_text(size = 6.3, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 6.8),
    legend.position = "none",
    strip.text = element_text(size = 8.2, face = "bold"),
    strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.45),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.45),
    plot.margin = margin(3, 4, 4, 4)
  )

make_metric_plot <- function(metric, strip_label, y_label, zero_reference = FALSE) {
  plot_data <- summary %>%
    filter(as.character(.data$Metric) == .env$metric) %>%
    mutate(Strip = .env$strip_label)

  plot <- ggplot(
    plot_data,
    aes(x = .data$Method_Label, y = .data$Mean, fill = .data$Method)
  ) +
    geom_col(width = 0.68, color = "black", linewidth = 0.3) +
    geom_errorbar(
      data = plot_data %>% filter(.data$N > 1),
      aes(ymin = .data$Mean - .data$SD, ymax = .data$Mean + .data$SD),
      width = 0.18,
      linewidth = 0.35,
      color = "black"
    ) +
    facet_wrap(~Strip, nrow = 1) +
    scale_fill_manual(values = method_colors) +
    scale_y_continuous(expand = expansion(mult = c(0.015, 0.08))) +
    labs(x = NULL, y = y_label) +
    metric_theme

  if (zero_reference) {
    plot <- plot + geom_hline(
      yintercept = 0,
      linetype = "dashed",
      color = "#D44C3C",
      linewidth = 0.5
    )
  }
  plot
}

make_group <- function(left_plot, right_plot, title) {
  heading <- ggdraw() +
    draw_label(title, x = 0.5, y = 0.5, size = 10, fontface = "bold")
  body <- plot_grid(left_plot, right_plot, nrow = 1, align = "h", axis = "tb")
  plot_grid(heading, body, ncol = 1, rel_heights = c(0.10, 0.90))
}

clustering_group <- make_group(
  make_metric_plot("ARI", "ARI", "Score"),
  make_metric_plot("NMI", "NMI", "Score"),
  "Clustering Performance"
)
mixing_group <- make_group(
  make_metric_plot("ASW_batch", "Raw ASW (Batch)", "Raw ASW", TRUE),
  make_metric_plot("iLISI", "iLISI", "iLISI"),
  "Batch Mixing"
)
biology_group <- make_group(
  make_metric_plot("ASW_cell", "ASW (Cell Type)", "ASW"),
  make_metric_plot("Graph_connectivity", "Graph Connectivity", "Score"),
  "Biological Structure Preservation"
)

panel_b_base <- plot_grid(
  clustering_group,
  mixing_group,
  biology_group,
  nrow = 1,
  rel_widths = c(1, 1, 1)
)

umap_data <- read.csv(umap_file, check.names = FALSE)
required_umap <- c(
  "cell_id", "batch", "celltype",
  "UMAP1_PCA", "UMAP2_PCA",
  "UMAP1_Harmony", "UMAP2_Harmony",
  "UMAP1_CONCORD", "UMAP2_CONCORD",
  "UMAP1_FederatedBatch", "UMAP2_FederatedBatch"
)
if (!all(required_umap %in% names(umap_data))) {
  stop("Figure 1 UMAP coordinate file is missing required columns")
}
if (nrow(umap_data) != 16382) {
  stop(sprintf("Expected 16,382 UMAP rows, found %d", nrow(umap_data)))
}

umap_methods <- c("PCA", "Harmony", "CONCORD", "FederatedBatch")

make_umap_panel <- function(color_field, palette, legend_title, legend_fraction) {
  plots <- list()
  for (method_name in umap_methods) {
    plot_data <- data.frame(
      UMAP1 = umap_data[[paste0("UMAP1_", method_name)]],
      UMAP2 = umap_data[[paste0("UMAP2_", method_name)]],
      Group = umap_data[[color_field]]
    )
    plots[[method_name]] <- ggplot(
      plot_data,
      aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Group)
    ) +
      geom_point(size = 0.10, alpha = 0.5) +
      scale_color_manual(values = palette) +
      labs(title = method_name, x = "UMAP 1", y = "UMAP 2") +
      theme_classic() +
      theme(
        plot.title = element_text(size = 10.5, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 8.5),
        axis.text = element_text(size = 7.5),
        legend.position = "none",
        aspect.ratio = 1,
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.45),
        plot.margin = margin(3, 4, 3, 4)
      )
  }

  legend_data <- data.frame(
    UMAP1 = umap_data$UMAP1_FederatedBatch,
    UMAP2 = umap_data$UMAP2_FederatedBatch,
    Group = umap_data[[color_field]]
  )
  legend_plot <- ggplot(
    legend_data,
    aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Group)
  ) +
    geom_point(size = 1.4, alpha = 0.8) +
    scale_color_manual(values = palette, name = legend_title) +
    guides(color = guide_legend(override.aes = list(size = 2.8, alpha = 0.8))) +
    theme_classic() +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 9.5, face = "bold"),
      legend.text = element_text(size = 7.7),
      legend.key.size = grid::unit(0.38, "cm")
    )

  legend <- get_legend(legend_plot)
  body <- plot_grid(plotlist = plots, nrow = 1, align = "h")
  centered_legend <- plot_grid(
    NULL,
    legend,
    NULL,
    ncol = 1,
    rel_heights = c((1 - legend_fraction) / 2, legend_fraction, (1 - legend_fraction) / 2)
  )
  plot_grid(body, centered_legend, nrow = 1, rel_widths = c(0.84, 0.16))
}

panel_c_base <- make_umap_panel("batch", BATCH_COLORS, "Batch", 0.50)
panel_d_base <- make_umap_panel("celltype", CELLTYPE_COLORS, "Cell Type", 0.72)

add_panel_label <- function(plot, label) {
  ggdraw() +
    draw_plot(plot, x = 0.018, y = 0.005, width = 0.982, height = 0.975) +
    draw_label(label, x = 0.002, y = 0.995, hjust = 0, vjust = 1,
               size = 13, fontface = "bold")
}

panel_b <- add_panel_label(panel_b_base, "B")
panel_c <- add_panel_label(panel_c_base, "C")
panel_d <- add_panel_label(panel_d_base, "D")

ggsave(
  file.path(output_dir, "Figure1B_final.png"),
  panel_b,
  width = 12,
  height = 3.6,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "Figure1C_final.png"),
  panel_c,
  width = 12,
  height = 2.8,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "Figure1D_final.png"),
  panel_d,
  width = 12,
  height = 2.8,
  dpi = 300,
  bg = "white"
)

combined_bcd <- plot_grid(
  panel_b,
  panel_c,
  panel_d,
  ncol = 1,
  rel_heights = c(3.6, 2.8, 2.8)
)

for (stem in c("Figure1_BCD_final", "Figure1_final")) {
  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    combined_bcd,
    width = 12,
    height = 9.2,
    dpi = 300,
    bg = "white"
  )
  ggsave(
    file.path(output_dir, paste0(stem, ".pdf")),
    combined_bcd,
    width = 12,
    height = 9.2,
    device = cairo_pdf,
    bg = "white"
  )
}

caption_summary <- summary %>%
  transmute(
    record_type = "metric_summary",
    method = as.character(.data$Method),
    comparator = NA_character_,
    metric = as.character(.data$Metric),
    n = .data$N,
    mean = .data$Mean,
    sd = .data$SD,
    mean_paired_difference = NA_real_,
    ci95_low = NA_real_,
    ci95_high = NA_real_,
    raw_p = NA_real_,
    holm_p = NA_real_,
    interpretation = ifelse(
      .data$Method == "PCA",
      "Deterministic embedding; no replicate error bar",
      "Mean and SD across seven matched training seeds"
    )
  )

paired_tests <- read.csv(tests_file, check.names = FALSE)
graph_tests <- paired_tests %>%
  filter(.data$metric == "Graph_connectivity") %>%
  mutate(
    comparator_label = recode(
      .data$comparator,
      "centralized_CONCORD" = "centralized CONCORD",
      "Harmony" = "Harmony"
    ),
    interpretation_text = ifelse(
      .data$comparator == "centralized_CONCORD",
      "FederatedBatch numerically lower; not significant after Holm correction",
      "FederatedBatch significantly higher after Holm correction"
    )
  ) %>%
  transmute(
    record_type = "paired_graph_connectivity_test",
    method = "FederatedBatch",
    comparator = .data$comparator_label,
    metric = "Graph_connectivity",
    n = .data$n_pairs,
    mean = NA_real_,
    sd = NA_real_,
    mean_paired_difference = .data$mean_paired_difference,
    ci95_low = .data$ci95_low,
    ci95_high = .data$ci95_high,
    raw_p = .data$paired_t_raw_p,
    holm_p = .data$paired_t_holm_p,
    interpretation = .data$interpretation_text
  )

caption_data <- bind_rows(caption_summary, graph_tests)
write.csv(
  caption_data,
  file.path(output_dir, "figure1_final_caption_data.csv"),
  row.names = FALSE
)

fb <- summary %>% filter(.data$Method == "FederatedBatch")
metric_value <- function(metric_name, column) {
  as.numeric(fb[as.character(fb$Metric) == metric_name, column])
}

report <- c(
  "# Figure 1 canonical-seven finalization",
  "",
  "Status: Figure 1B-D COMPLETE. Figure 1A is intentionally omitted for manual preparation.",
  "",
  "## Panel B",
  "",
  "- Clustering Performance: ARI, NMI.",
  "- Batch Mixing: Raw ASW (Batch), iLISI.",
  "- Biological Structure Preservation: ASW (Cell Type), Graph Connectivity.",
  "- Stochastic-method error bars are mean +/- SD across seeds 218, 394, 542, 639, 788, 814, and 880.",
  "- PCA is deterministic (n=1) and has no error bar.",
  "- The raw ASW(batch) panel retains a zero reference line and is not labeled as a larger-is-better metric.",
  "- The only manuscript-facing name for the final method is FederatedBatch.",
  "",
  "## Panels C-D",
  "",
  "- Representative training seed: 218.",
  "- UMAP parameters: n_neighbors=15, min_dist=0.1, Euclidean metric, random_state=42.",
  "- Panels C and D are rendered from exactly the same saved coordinate columns for each method.",
  "- FederatedBatch uses the completed canonical-seven seed-218 embedding.",
  "",
  "## Caption-support values",
  "",
  sprintf("- ARI: %.3f +/- %.3f.", metric_value("ARI", "Mean"), metric_value("ARI", "SD")),
  sprintf("- NMI: %.3f +/- %.3f.", metric_value("NMI", "Mean"), metric_value("NMI", "SD")),
  sprintf("- Raw ASW(batch): %.4f +/- %.4f.", metric_value("ASW_batch", "Mean"), metric_value("ASW_batch", "SD")),
  sprintf("- ASW(cell type): %.4f +/- %.4f.", metric_value("ASW_cell", "Mean"), metric_value("ASW_cell", "SD")),
  sprintf("- Graph connectivity: %.4f +/- %.4f.", metric_value("Graph_connectivity", "Mean"), metric_value("Graph_connectivity", "SD")),
  sprintf("- iLISI: %.4f +/- %.4f.", metric_value("iLISI", "Mean"), metric_value("iLISI", "SD")),
  "- Graph connectivity versus centralized CONCORD: numerically lower but not significant after Holm correction (paired t-test).",
  "- Graph connectivity versus Harmony: significantly higher after Holm correction (paired t-test).",
  "- No claim of global superiority over Harmony or centralized CONCORD is supported.",
  "",
  "## Files",
  "",
  "`Figure1_final.*` and `Figure1_BCD_final.*` are identical B-D assemblies that intentionally leave Panel A for manual composition. Historical Figure 1 files were not overwritten."
)
writeLines(report, file.path(output_dir, "FIGURE1_FINALIZATION_REPORT.md"))

cat(sprintf("Figure 1B-D finalization complete: %s\n", output_dir))
