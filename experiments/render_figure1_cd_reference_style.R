#!/usr/bin/env Rscript

# Exact visual style of the legacy Figure 1 UMAP panels, applied to finalized
# canonical-seven embeddings and reference-protocol visualization coordinates.

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
})

source("external_artifacts/figure1/figures/figure1/r/color_config.R")

input_file <- paste0(
  "results/final_figure1_canonical7/reference_style_cd/",
  "figure1_umap_coordinates_reference_style.csv"
)
output_dir <- "results/final_figure1_canonical7/reference_style_cd"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

options(pdf.useDingbats = FALSE, bitmapType = "cairo")

umap_data <- read.csv(input_file, check.names = FALSE)
methods <- c("PCA", "Harmony", "CONCORD", "FederatedBatch")

make_panel <- function(color_field, palette, legend_title, panel_label, legend_fraction) {
  plots <- lapply(methods, function(method_name) {
    plot_data <- data.frame(
      UMAP1 = umap_data[[paste0("UMAP1_", method_name)]],
      UMAP2 = umap_data[[paste0("UMAP2_", method_name)]],
      Group = umap_data[[color_field]]
    )
    ggplot(plot_data, aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Group)) +
      geom_point(size = 0.10, alpha = 0.50) +
      scale_color_manual(values = palette, drop = FALSE) +
      labs(title = method_name, x = "UMAP 1", y = "UMAP 2") +
      theme_classic(base_family = "sans") +
      theme(
        plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 9),
        axis.text = element_text(size = 8, color = "#4D4D4D"),
        legend.position = "none",
        aspect.ratio = 1,
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
        plot.margin = margin(3, 5, 3, 5)
      )
  })

  legend_data <- data.frame(
    UMAP1 = umap_data$UMAP1_FederatedBatch,
    UMAP2 = umap_data$UMAP2_FederatedBatch,
    Group = umap_data[[color_field]]
  )
  legend_plot <- ggplot(
    legend_data,
    aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Group)
  ) +
    geom_point(size = 1.5, alpha = 0.8) +
    scale_color_manual(values = palette, name = legend_title, drop = FALSE) +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 0.8))) +
    theme_classic(base_family = "sans") +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 8),
      legend.key.size = grid::unit(0.4, "cm"),
      legend.margin = margin(0, 0, 0, 0)
    )
  legend <- get_legend(legend_plot)

  plot_body <- plot_grid(plotlist = plots, nrow = 1, align = "h", axis = "tb")
  centered_legend <- plot_grid(
    NULL, legend, NULL,
    ncol = 1,
    rel_heights = c((1 - legend_fraction) / 2, legend_fraction, (1 - legend_fraction) / 2)
  )
  body <- plot_grid(plot_body, centered_legend, nrow = 1, rel_widths = c(0.84, 0.16))

  ggdraw() +
    draw_plot(body, x = 0.018, y = 0.003, width = 0.982, height = 0.985) +
    draw_label(panel_label, x = 0.001, y = 0.997, hjust = 0, vjust = 1,
               size = 13, fontface = "bold", fontfamily = "sans")
}

panel_c <- make_panel("batch", BATCH_COLORS, "Batch", "C", 0.50)
panel_d <- make_panel("celltype", CELLTYPE_COLORS, "Cell Type", "D", 0.70)

save_plot <- function(stem, plot, width, height) {
  ggsave(file.path(output_dir, paste0(stem, ".png")), plot,
         width = width, height = height, dpi = 600, bg = "white")
  ggsave(file.path(output_dir, paste0(stem, ".pdf")), plot,
         width = width, height = height, device = cairo_pdf, bg = "white")
  if (requireNamespace("svglite", quietly = TRUE)) {
    ggsave(file.path(output_dir, paste0(stem, ".svg")), plot,
           width = width, height = height, device = svglite::svglite, bg = "white")
  }
}

save_plot("Figure1C_reference_style", panel_c, 10, 2.5)
save_plot("Figure1D_reference_style", panel_d, 10, 2.5)
combined <- plot_grid(panel_c, panel_d, ncol = 1, rel_heights = c(1, 1))
save_plot("Figure1_CD_reference_style", combined, 10, 5)

cat(sprintf("Reference-style Figure 1C-D written to %s\n", output_dir))
