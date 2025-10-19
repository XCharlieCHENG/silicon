#' Plotting Module for LLM Annotation Analysis
#'
#' This module provides functions for creating coefficient plots to visualize
#' model and prompt comparisons.

library(ggplot2)
library(dplyr)
library(stringr)

#' Create coefficient plot for model comparison
#'
#' @param results Model results from run_model_comparison_logit
#' @param output_file Optional file path to save the plot
#' @return ggplot object
plot_model_comparison <- function(results, output_file = NULL) {
  # Extract coefficients, excluding the intercept
  coef_data <- results$coefficients %>%
    filter(term != "(Intercept)") %>%
    mutate(
      # Extract treatment number from term
      treatment_num = as.numeric(str_extract(term, "\\d+")),
      # Calculate confidence intervals (95%)
      ci_lower = estimate - 1.96 * std.error,
      ci_upper = estimate + 1.96 * std.error
    )

  # Define treatment order and labels
  treatment_order <- c("4.treatment", "8.treatment", "9.treatment", "38.treatment", "39.treatment", "57.treatment")
  treatment_labels <- c(
    "4" = "GPT-4 Turbo",
    "8" = "Gemini 1.5 Pro",
    "9" = "Claude 3.5 Sonnet",
    "38" = "o3-mini",
    "39" = "LLaMA 3.3 70B",
    "57" = "DeepSeek-R1"
  )

  # Create factor for ordering
  coef_data <- coef_data %>%
    mutate(
      treatment_label = treatment_labels[as.character(treatment_num)],
      treatment_label = factor(treatment_label, levels = treatment_labels)
    )

  # Create the plot
  p <- ggplot(coef_data, aes(x = treatment_label, y = estimate)) +
    geom_hline(yintercept = 0, color = "black", linewidth = 1) +
    geom_point(size = 3, color = "black") +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                  width = 0.2, color = "black", linewidth = 1) +
    coord_flip() +
    labs(
      x = "",
      y = "Performance Difference Relative to GPT-4o",
      title = ""
    ) +
    theme_bw() +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10, angle = 0),
      axis.title = element_text(size = 12),
      plot.margin = unit(c(1, 1, 1, 1), "cm")
    ) +
    scale_y_continuous(position = "right")

  # Save plot if output_file is provided
  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 10, height = 6, dpi = 300)
  }

  return(p)
}

#' Create coefficient plot for prompt comparison
#'
#' @param results Model results from run_prompt_comparison_lm
#' @param output_file Optional file path to save the plot
#' @return ggplot object
plot_prompt_comparison <- function(results, output_file = NULL) {
  # Extract coefficients, excluding the intercept
  coef_data <- results$coefficients %>%
    filter(term != "(Intercept)") %>%
    mutate(
      # Extract treatment number from term
      treatment_num = as.numeric(str_extract(term, "\\d+")),
      # Calculate confidence intervals (95%)
      ci_lower = estimate - 1.96 * std.error,
      ci_upper = estimate + 1.96 * std.error
    )

  # Define treatment order and labels for prompt comparison
  treatment_labels <- c(
    "19" = "System Role; Persona",
    "20" = "System Role; CoT",
    "18" = "User Role; Base",
    "27" = "User Role; Persona",
    "28" = "User Role; CoT"
  )

  # Create factor for ordering
  coef_data <- coef_data %>%
    mutate(
      treatment_label = treatment_labels[as.character(treatment_num)],
      treatment_label = factor(treatment_label, levels = treatment_labels)
    )

  # Create the plot
  p <- ggplot(coef_data, aes(x = treatment_label, y = estimate)) +
    geom_hline(yintercept = 0, color = "black", linewidth = 1) +
    geom_point(size = 3, color = "black") +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                  width = 0.2, color = "black", linewidth = 1) +
    coord_flip() +
    labs(
      x = "",
      y = "Performance Difference Relative to 'System Role; Base'",
      title = ""
    ) +
    theme_bw() +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10, angle = 0),
      axis.title = element_text(size = 12),
      plot.margin = unit(c(1, 1, 1, 1), "cm")
    ) +
    scale_y_continuous(position = "right")

  # Save plot if output_file is provided
  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 10, height = 6, dpi = 300)
  }

  return(p)
}

#' Create a combined plot with both model and prompt comparisons
#'
#' @param model_results Results from model comparison
#' @param prompt_results Results from prompt comparison
#' @param output_file Optional file path to save the combined plot
#' @return List of ggplot objects
create_combined_plots <- function(model_results, prompt_results, output_file = NULL) {
  plots <- list(
    model_comparison = plot_model_comparison(model_results),
    prompt_comparison = plot_prompt_comparison(prompt_results)
  )

  # If output_file is provided, save both plots
  if (!is.null(output_file)) {
    # Extract directory and base name
    dir_name <- dirname(output_file)
    base_name <- tools::file_path_sans_ext(basename(output_file))
    ext <- tools::file_ext(output_file)

    # Save individual plots
    ggsave(file.path(dir_name, paste0(base_name, "_model.", ext)),
           plots$model_comparison, width = 10, height = 6, dpi = 300)
    ggsave(file.path(dir_name, paste0(base_name, "_prompt.", ext)),
           plots$prompt_comparison, width = 10, height = 6, dpi = 300)
  }

  return(plots)
}


