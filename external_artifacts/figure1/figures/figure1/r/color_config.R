# 统一配色方案配置文件
# 用于所有Figure 1-4的可视化
# 创建时间: 2026-03-18

# ============================================================================
# 方法配色（4个主要方法）
# ============================================================================

METHOD_COLORS <- c(
  "PCA" = rgb(183, 181, 160, maxColorValue = 255),              # 灰绿色 - Baseline
  "Harmony" = rgb(68, 117, 122, maxColorValue = 255),           # 青蓝色 - Batch correction
  "CONCORD" = rgb(69, 42, 61, maxColorValue = 255),             # 深紫色 - Centralized CONCORD
  "Centralized" = rgb(69, 42, 61, maxColorValue = 255),         # 深紫色 - 别名
  "Centralized_CONCORD" = rgb(69, 42, 61, maxColorValue = 255), # 深紫色 - 别名
  "FedCONCORD" = rgb(212, 76, 60, maxColorValue = 255),         # 橙红色 - 我们的方法（重点）
  "FedCONCORD_3clients" = rgb(212, 76, 60, maxColorValue = 255),# 橙红色 - 别名
  "FedscGen" = rgb(68, 117, 122, maxColorValue = 255)           # 青蓝色 - 对比方法
)

# ============================================================================
# 批次配色 - 使用配色系列的扩展
# ============================================================================

BATCH_COLORS <- c(
  # Pancreas 数据集批次 (9个)
  "celseq" = rgb(183, 181, 160, maxColorValue = 255),      # 灰绿色
  "celseq2" = rgb(68, 117, 122, maxColorValue = 255),      # 青蓝色
  "fluidigmc1" = rgb(69, 42, 61, maxColorValue = 255),     # 深紫色
  "inDrop1" = rgb(212, 76, 60, maxColorValue = 255),       # 橙红色
  "inDrop2" = rgb(221, 108, 76, maxColorValue = 255),      # 浅橙色
  "inDrop3" = rgb(229, 133, 93, maxColorValue = 255),      # 淡橙色
  "inDrop4" = rgb(238, 213, 183, maxColorValue = 255),     # 米黄色
  "smarter" = rgb(150, 100, 120, maxColorValue = 255),     # 中紫色
  "smartseq2" = rgb(100, 140, 145, maxColorValue = 255),   # 中青色
  
  # Immune 数据集批次 (10个)
  "10X" = rgb(212, 76, 60, maxColorValue = 255),           # 橙红色
  "Freytag" = rgb(221, 108, 76, maxColorValue = 255),      # 浅橙色
  "Oetjen_A" = rgb(229, 133, 93, maxColorValue = 255),     # 淡橙色
  "Oetjen_P" = rgb(238, 213, 183, maxColorValue = 255),    # 米黄色
  "Oetjen_U" = rgb(183, 181, 160, maxColorValue = 255),    # 灰绿色
  "Sun_sample1_CS" = rgb(150, 170, 165, maxColorValue = 255), # 浅灰绿
  "Sun_sample2_KC" = rgb(68, 117, 122, maxColorValue = 255),  # 青蓝色
  "Sun_sample3_TB" = rgb(100, 140, 145, maxColorValue = 255), # 中青色
  "Sun_sample4_TC" = rgb(69, 42, 61, maxColorValue = 255),    # 深紫色
  "Villani" = rgb(120, 80, 100, maxColorValue = 255),         # 中紫色
  
  # Lung 数据集批次 (16个)
  "1" = rgb(212, 76, 60, maxColorValue = 255),             # 橙红色
  "2" = rgb(221, 108, 76, maxColorValue = 255),            # 浅橙色
  "3" = rgb(229, 133, 93, maxColorValue = 255),            # 淡橙色
  "4" = rgb(238, 213, 183, maxColorValue = 255),           # 米黄色
  "5" = rgb(183, 181, 160, maxColorValue = 255),           # 灰绿色
  "6" = rgb(150, 170, 165, maxColorValue = 255),           # 浅灰绿
  "A1" = rgb(68, 117, 122, maxColorValue = 255),           # 青蓝色
  "A2" = rgb(80, 130, 135, maxColorValue = 255),           # 浅青蓝
  "A3" = rgb(100, 140, 145, maxColorValue = 255),          # 中青色
  "A4" = rgb(69, 42, 61, maxColorValue = 255),             # 深紫色
  "A5" = rgb(90, 60, 80, maxColorValue = 255),             # 中深紫
  "A6" = rgb(120, 80, 100, maxColorValue = 255),           # 中紫色
  "B1" = rgb(150, 100, 120, maxColorValue = 255),          # 浅紫色
  "B2" = rgb(180, 130, 150, maxColorValue = 255),          # 淡紫色
  "B3" = rgb(200, 150, 170, maxColorValue = 255),          # 更淡紫色
  "B4" = rgb(220, 180, 190, maxColorValue = 255)           # 极淡紫色
)

# ============================================================================
# 细胞类型配色 - 使用统一的配色系列渐变（与 Figure 1 一致）
# ============================================================================

CELLTYPE_COLORS <- c(
  # Pancreas 数据集 (14 types) - 原始配色
  "acinar" = rgb(212, 76, 60, maxColorValue = 255),           # 橙红色
  "activated_stellate" = rgb(221, 108, 76, maxColorValue = 255), # 浅橙色
  "alpha" = rgb(229, 133, 93, maxColorValue = 255),           # 淡橙色
  "beta" = rgb(238, 213, 183, maxColorValue = 255),           # 米黄色
  "delta" = rgb(183, 181, 160, maxColorValue = 255),          # 灰绿色
  "ductal" = rgb(150, 170, 165, maxColorValue = 255),         # 浅灰绿
  "endothelial" = rgb(68, 117, 122, maxColorValue = 255),     # 青蓝色
  "epsilon" = rgb(80, 130, 135, maxColorValue = 255),         # 浅青蓝
  "gamma" = rgb(100, 140, 145, maxColorValue = 255),          # 中青色
  "macrophage" = rgb(69, 42, 61, maxColorValue = 255),        # 深紫色
  "mast" = rgb(90, 60, 80, maxColorValue = 255),              # 中深紫
  "quiescent_stellate" = rgb(120, 80, 100, maxColorValue = 255), # 中紫色
  "schwann" = rgb(150, 100, 120, maxColorValue = 255),        # 浅紫色
  "t_cell" = rgb(180, 130, 150, maxColorValue = 255),         # 淡紫色
  
  # Immune 数据集 (16 types) - 使用相同的配色系列
  "CD10+ B cells" = rgb(212, 76, 60, maxColorValue = 255),    # 橙红色
  "CD14+ Monocytes" = rgb(221, 108, 76, maxColorValue = 255), # 浅橙色
  "CD16+ Monocytes" = rgb(229, 133, 93, maxColorValue = 255), # 淡橙色
  "CD20+ B cells" = rgb(238, 213, 183, maxColorValue = 255),  # 米黄色
  "CD4+ T cells" = rgb(183, 181, 160, maxColorValue = 255),   # 灰绿色
  "CD8+ T cells" = rgb(150, 170, 165, maxColorValue = 255),   # 浅灰绿
  "Erythrocytes" = rgb(68, 117, 122, maxColorValue = 255),    # 青蓝色
  "Erythroid progenitors" = rgb(80, 130, 135, maxColorValue = 255), # 浅青蓝
  "HSPCs" = rgb(100, 140, 145, maxColorValue = 255),          # 中青色
  "Megakaryocyte progenitors" = rgb(69, 42, 61, maxColorValue = 255), # 深紫色
  "Monocyte progenitors" = rgb(90, 60, 80, maxColorValue = 255), # 中深紫
  "Monocyte-derived dendritic cells" = rgb(120, 80, 100, maxColorValue = 255), # 中紫色
  "NK cells" = rgb(150, 100, 120, maxColorValue = 255),       # 浅紫色
  "NKT cells" = rgb(180, 130, 150, maxColorValue = 255),      # 淡紫色
  "Plasma cells" = rgb(200, 150, 170, maxColorValue = 255),   # 更淡紫色
  "Plasmacytoid dendritic cells" = rgb(220, 180, 190, maxColorValue = 255), # 极淡紫色
  
  # Lung 数据集 (17 types) - 使用相同的配色系列
  "B cell" = rgb(212, 76, 60, maxColorValue = 255),           # 橙红色
  "Basal 1" = rgb(221, 108, 76, maxColorValue = 255),         # 浅橙色
  "Basal 2" = rgb(229, 133, 93, maxColorValue = 255),         # 淡橙色
  "Ciliated" = rgb(238, 213, 183, maxColorValue = 255),       # 米黄色
  "Dendritic cell" = rgb(183, 181, 160, maxColorValue = 255), # 灰绿色
  "Endothelium" = rgb(150, 170, 165, maxColorValue = 255),    # 浅灰绿
  "Fibroblast" = rgb(68, 117, 122, maxColorValue = 255),      # 青蓝色
  "Ionocytes" = rgb(80, 130, 135, maxColorValue = 255),       # 浅青蓝
  "Lymphatic" = rgb(100, 140, 145, maxColorValue = 255),      # 中青色
  "Macrophage" = rgb(69, 42, 61, maxColorValue = 255),        # 深紫色
  "Mast cell" = rgb(90, 60, 80, maxColorValue = 255),         # 中深紫
  "Neutrophil_CD14_high" = rgb(120, 80, 100, maxColorValue = 255), # 中紫色
  "Neutrophils_IL1R2" = rgb(150, 100, 120, maxColorValue = 255), # 浅紫色
  "Secretory" = rgb(180, 130, 150, maxColorValue = 255),      # 淡紫色
  "T/NK cell" = rgb(200, 150, 170, maxColorValue = 255),      # 更淡紫色
  "Type 1" = rgb(220, 180, 190, maxColorValue = 255),         # 极淡紫色
  "Type 2" = rgb(235, 210, 215, maxColorValue = 255)          # 最淡紫色
)

# ============================================================================
# 辅助函数
# ============================================================================

# 获取方法颜色
get_method_color <- function(method_name) {
  if (method_name %in% names(METHOD_COLORS)) {
    return(METHOD_COLORS[method_name])
  } else {
    warning(paste("Unknown method:", method_name, "- using default color"))
    return("#999999")  # 默认灰色
  }
}

# 获取批次颜色
get_batch_color <- function(batch_name) {
  if (batch_name %in% names(BATCH_COLORS)) {
    return(BATCH_COLORS[batch_name])
  } else {
    warning(paste("Unknown batch:", batch_name, "- using default color"))
    return("#999999")  # 默认灰色
  }
}

# 获取细胞类型颜色
get_celltype_color <- function(celltype_name) {
  if (celltype_name %in% names(CELLTYPE_COLORS)) {
    return(CELLTYPE_COLORS[celltype_name])
  } else {
    warning(paste("Unknown celltype:", celltype_name, "- using default color"))
    return("#999999")  # 默认灰色
  }
}

# ============================================================================
# 配色信息打印
# ============================================================================

print_color_scheme <- function() {
  cat("================================================================================\n")
  cat("统一配色方案\n")
  cat("================================================================================\n\n")
  
  cat("方法配色:\n")
  for (name in names(METHOD_COLORS)) {
    cat(sprintf("  %s: %s\n", name, METHOD_COLORS[name]))
  }
  
  cat("\n批次配色:\n")
  for (name in names(BATCH_COLORS)) {
    cat(sprintf("  %s: %s\n", name, BATCH_COLORS[name]))
  }
  
  cat("\n细胞类型配色:\n")
  for (name in names(CELLTYPE_COLORS)) {
    cat(sprintf("  %s: %s\n", name, CELLTYPE_COLORS[name]))
  }
  
  cat("\n================================================================================\n")
}

# 使用示例:
# source("scripts/visualization_v2/common/color_config.R")
# print_color_scheme()
